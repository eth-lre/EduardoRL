#!/bin/bash
#SBATCH --job-name=merge-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=240000
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --output=logs/merge-%j.out
#SBATCH --error=logs/merge-%j.err
#
# Merge an FSDP-sharded veRL checkpoint into HuggingFace format:
#   ${WORK_DIR}/verl_ckpts/<experiment>/global_step_<step>/actor
#   -> ${WORK_DIR}/eval_checkpoints/<experiment>/step_<step>
#
# Usage:
#   sbatch scripts/merge_checkpoint.sh <experiment> <step>

set -euo pipefail

: "${WORK_DIR:=${SCRATCH:-$HOME/scratch}/eduardo}"

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

EXPERIMENT="${1:?Usage: sbatch $0 <experiment> <step>}"
STEP="${2:?Usage: sbatch $0 <experiment> <step>}"

LOCAL_DIR="${WORK_DIR}/verl_ckpts/${EXPERIMENT}/global_step_${STEP}/actor"
TARGET_DIR="${WORK_DIR}/eval_checkpoints/${EXPERIMENT}/step_${STEP}"

echo "Merging FSDP checkpoint:"
echo "  Source: ${LOCAL_DIR}"
echo "  Target: ${TARGET_DIR}"

mkdir -p "${TARGET_DIR}"

# ── Container (pyxis) ───────────────────────────────────────────────────────
# Image built from the repo's Dockerfile. Set CONTAINER_ARGS="" to run on the host.
: "${CONTAINER_IMAGE:=${WORK_DIR}/eduardo.sqsh}"
: "${CONTAINER_MOUNTS:=${HOME}:${HOME},${WORK_DIR}:${WORK_DIR}}"
: "${CONTAINER_ARGS=--container-image=${CONTAINER_IMAGE} --container-mounts=${CONTAINER_MOUNTS}}"

srun ${CONTAINER_ARGS} \
    bash -c "
        set -e
        echo 'Testing write access to ${TARGET_DIR}...'
        touch '${TARGET_DIR}/.write_test' && rm '${TARGET_DIR}/.write_test'
        echo 'Write access OK'
        
        python -m verl.model_merger merge --backend fsdp \
            --local_dir '${LOCAL_DIR}' \
            --target_dir '${TARGET_DIR}'
        
        echo 'Verifying output...'
        ls -la '${TARGET_DIR}'
    "

echo "Done. Merged model saved to: ${TARGET_DIR}"
