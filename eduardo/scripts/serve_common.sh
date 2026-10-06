#!/usr/bin/env bash
# Shared helper for the launch_*_server.sh scripts: submits a persistent
# single-node SLURM job named serve_<SERVED_NAME> running `vllm serve` on port
# 8080. Training/eval scripts locate its node via squeue by job name.
#
# Env: WORK_DIR, CONTAINER_IMAGE, CONTAINER_MOUNTS, CONTAINER_ARGS,
#      SLURM_PARTITION, SLURM_ACCOUNT, SLURM_TIME, GPUS_PER_NODE,
#      VLLM_API_KEY (optional; clients must pass it as SERVING_API_KEY)

: "${WORK_DIR:=${SCRATCH:-$HOME/scratch}/eduardo}"
: "${CONTAINER_IMAGE:=${WORK_DIR}/eduardo.sqsh}"
: "${CONTAINER_MOUNTS:=${HOME}:${HOME},${WORK_DIR}:${WORK_DIR}}"
: "${CONTAINER_ARGS=--container-image=${CONTAINER_IMAGE} --container-mounts=${CONTAINER_MOUNTS}}"
: "${SLURM_PARTITION:=normal}"
: "${SLURM_TIME:=11:59:00}"
: "${GPUS_PER_NODE:=4}"

# ── Caches on scratch, never in $HOME ───────────────────────────────────────
# Exported here because sbatch forwards the submitting environment (--export=ALL).
export XDG_CACHE_HOME="${WORK_DIR}/.cache"
export HF_HOME="${WORK_DIR}/huggingface"
export VLLM_CACHE_ROOT="${XDG_CACHE_HOME}/vllm"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME}/inductor"
mkdir -p "${HF_HOME}" "${VLLM_CACHE_ROOT}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}" \
         "${TORCHINDUCTOR_CACHE_DIR}" "${WORK_DIR}/logs"

# submit_vllm_server <served-name> <model> <extra vllm serve args...>
submit_vllm_server() {
  local served_name="$1" model="$2"
  shift 2
  local api_key_args=""
  [[ -n "${VLLM_API_KEY:-}" ]] && api_key_args="--api-key ${VLLM_API_KEY}"
  local account_args=()
  [[ -n "${SLURM_ACCOUNT:-}" ]] && account_args=(--account="${SLURM_ACCOUNT}")
  # Slashes would break the job-name lookup pattern.
  local job_name="serve_${served_name//\//_}"

  sbatch --job-name="${job_name}" \
         --nodes=1 --ntasks-per-node=1 \
         --gpus-per-node="${GPUS_PER_NODE}" \
         --partition="${SLURM_PARTITION}" \
         --time="${SLURM_TIME}" \
         "${account_args[@]}" \
         --output="${WORK_DIR}/logs/${job_name}-%j.out" \
         --export=ALL \
         --wrap="srun ${CONTAINER_ARGS} vllm serve ${model} \
           --host 0.0.0.0 \
           --port 8080 \
           --served-model-name ${served_name} \
           ${api_key_args} $*"
}
