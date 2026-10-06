#!/bin/bash
#SBATCH --job-name=eduardo-verl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --partition=normal
#SBATCH --time=11:59:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# SLURM submit script for a single Eduardo training run (inside the container
# built from the repo's Dockerfile). trainer.nnodes, n_gpus_per_node and
# fsdp_size are derived from the allocation in start_training.sh.
#
# Required env vars:
#   SERVING_API_KEY     bearer token of the vLLM serving endpoints
#   WANDB_API_KEY       for W&B login
#   WANDB_ENTITY        W&B team name
#   EXPERIMENT          unique experiment name (W&B run + ckpt dir)
#   WORK_DIR            scratch dir for ckpts/logs
#   EDUARDO_RECIPE_DIR  path to eduardo/ in the checkout
#
# Optional env vars:
#   MODEL_PATH          policy model; overrides actor_rollout_ref.model.path
#   GEMINI_API_KEY      enables the 1-5 pedagogy/correctness judge at validation
#
# Usage:
#   sbatch scripts/train_slurm.sh eduardo_base
#   sbatch --export=ALL,EXPERIMENT=lr1e6-n16 scripts/train_slurm.sh eduardo_base \
#       actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.rollout.n=16
#   sbatch --export=ALL,EXPERIMENT=qwen4b,MODEL_PATH=Qwen/Qwen3.5-4B \
#       scripts/train_slurm.sh eduardo_base
#   sbatch --nodes=4 --export=ALL,EXPERIMENT=n4 scripts/train_slurm.sh eduardo_base

set -euo pipefail

# ── Required env ────────────────────────────────────────────────────────────
: "${SERVING_API_KEY:?set SERVING_API_KEY before sbatch}"
: "${WANDB_API_KEY:?set WANDB_API_KEY before sbatch}"
: "${WANDB_ENTITY:?set WANDB_ENTITY before sbatch (team name)}"
: "${EXPERIMENT:?set EXPERIMENT=<unique-name> before sbatch}"
: "${WORK_DIR:=${SCRATCH:-$HOME/scratch}/eduardo}"
: "${EDUARDO_RECIPE_DIR:=$HOME/eduardo/eduardo}"
export WORK_DIR EDUARDO_RECIPE_DIR EXPERIMENT

# ── Container (pyxis) ───────────────────────────────────────────────────────
# Image built from the repo's Dockerfile. Set CONTAINER_ARGS="" to run on the host.
: "${CONTAINER_IMAGE:=${WORK_DIR}/eduardo.sqsh}"
: "${CONTAINER_MOUNTS:=${HOME}:${HOME},${WORK_DIR}:${WORK_DIR}}"
: "${CONTAINER_ARGS=--container-image=${CONTAINER_IMAGE} --container-mounts=${CONTAINER_MOUNTS}}"

# ── Optional model override ─────────────────────────────────────────────────
# An exported-but-empty MODEL_PATH would override the config default with "".
if [[ -z "${MODEL_PATH:-}" ]]; then
  unset MODEL_PATH || true
  echo "[train_slurm] MODEL_PATH unset -> using actor_rollout_ref.model.path from configs/user.yaml"
else
  export MODEL_PATH
  echo "[train_slurm] MODEL_PATH=${MODEL_PATH}"
fi

CONFIG_NAME="${1:-eduardo_base}"
shift || true
EXTRA_ARGS=("$@")

# ── Resolve serving endpoint nodes from SLURM job list ───────────────────────
# The serving jobs are named serve_<served-model-name> (launch_*_server.sh).
resolve_serving_node() {
  local prefix="$1"
  local node
  node=$(squeue --user="${USER}" --noheader --format="%j %N" \
         | grep "^${prefix}" | head -1 | awk '{print $2}')
  if [[ -z "${node}" ]]; then
    echo "[train_slurm] ERROR: no running SLURM job matching '${prefix}*' found. Launch it with scripts/launch_*_server.sh first." >&2
    exit 1
  fi
  echo "${node}"
}

STUDENT_MODEL_NAME="Llama-3.1-8B-Instruct-${USER}"
JUDGE_MODEL_NAME="Qwen3.6-27B-${USER}"

STUDENT_NODE=$(resolve_serving_node "serve_${STUDENT_MODEL_NAME}")
JUDGE_NODE=$(resolve_serving_node "serve_${JUDGE_MODEL_NAME}")

export STUDENT_API_BASE="http://${STUDENT_NODE}:8080/v1"
export JUDGE_API_BASE="http://${JUDGE_NODE}:8080/v1"

echo "[train_slurm] Student endpoint: ${STUDENT_API_BASE} (node=${STUDENT_NODE})"
echo "[train_slurm] Judge endpoint:   ${JUDGE_API_BASE} (node=${JUDGE_NODE})"

# ── Sanity-check serving endpoints (wait up to 10 min) ──────────────────────
check_endpoint() {
  local model="$1"
  local api_base="$2"
  local max_wait=600
  local interval=30
  local elapsed=0
  local http

  echo "[train_slurm] Waiting for ${model} at ${api_base} (max ${max_wait}s, checking every ${interval}s)..."

  while [[ ${elapsed} -lt ${max_wait} ]]; do
    http=$(curl -s -o /dev/null -w "%{http_code}" \
      -X POST "${api_base}/chat/completions" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer ${SERVING_API_KEY}" \
      -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" \
      || echo "000")

    if [[ "${http}" == "200" ]]; then
      echo "[train_slurm]   ${model} OK (${elapsed}s elapsed)"
      return 0
    fi

    echo "[train_slurm]   ${model} not ready (http=${http}), retrying in ${interval}s... (${elapsed}s elapsed)"
    sleep ${interval}
    elapsed=$((elapsed + interval))
  done

  echo "[train_slurm] ERROR: ${model} not ready after ${max_wait}s; start it with scripts/launch_*_server.sh first." >&2
  exit 1
}
check_endpoint "${STUDENT_MODEL_NAME}" "${STUDENT_API_BASE}"
check_endpoint "${JUDGE_MODEL_NAME}" "${JUDGE_API_BASE}"

# Short Ray tmpdir to stay under the AF_UNIX socket path limit (107 bytes).
export RAY_TMPDIR="/tmp/ray_${USER}"
mkdir -p "${WORK_DIR}/output" \
         "${WORK_DIR}/verl_ckpts" \
         "${RAY_TMPDIR}"

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

# ── GH200 environment ───────────────────────────────────────────────────────
export PYTHONPATH="${EDUARDO_RECIPE_DIR}/..:${PYTHONPATH:-}"
export VLLM_USE_V1=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES
unset VLLM_ATTENTION_BACKEND
ulimit -c 0

# ── Launch inside the pre-built squashfs environment ────────────────────────
srun --nodes="${SLURM_NNODES}" \
     --ntasks-per-node=1 \
     --export=ALL \
     ${CONTAINER_ARGS} \
     bash -lc "cd ${EDUARDO_RECIPE_DIR} && pip install hf_transfer qwen-vl-utils==0.0.14 && ./scripts/start_training.sh ${CONFIG_NAME} ${EXTRA_ARGS[*]:-}"
