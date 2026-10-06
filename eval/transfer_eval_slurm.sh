#!/bin/bash
#SBATCH --job-name=eduardo-transfer-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --partition=normal
#SBATCH --time=3:59:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Offline Δ_transfer, Δ_same and leak_rate on test.parquet for one checkpoint
# (eval/run_env_test_eval.py). Requires a running student server, and a judge
# server unless --gemini is passed.
#
# Env vars:
#   SERVING_API_KEY     bearer token of the vLLM serving endpoints
#   EVAL_MODEL          checkpoint path (or HF id) of the tutor
#   GEMINI_API_KEY      needed for --gemini and for the 1-5 eval judge
#   WORK_DIR            scratch dir; also where test.parquet lives
#   EDUARDO_RECIPE_DIR  path to eduardo/ in the checkout
#   TEST_PARQUET        optional override of ${WORK_DIR}/data/eduardo/test.parquet
#
# Usage:
#   sbatch --export=ALL,EVAL_MODEL=${WORK_DIR}/eval_checkpoints/arm/step_75 \
#       eval/transfer_eval_slurm.sh --gemini --enable-thinking --num-problems 100

set -euo pipefail

# ── Required env ────────────────────────────────────────────────────────────
: "${SERVING_API_KEY:?set SERVING_API_KEY before sbatch}"
: "${EVAL_MODEL:?set EVAL_MODEL=<hf-id-or-local-path> before sbatch}"
: "${WORK_DIR:=${SCRATCH:-$HOME/scratch}/eduardo}"
: "${EDUARDO_RECIPE_DIR:=$HOME/eduardo/eduardo}"
# Held-out test split; val.parquet is used for checkpoint selection during training.
: "${TEST_PARQUET:=${VAL_PARQUET:-${WORK_DIR}/data/eduardo/test.parquet}}"
export WORK_DIR EDUARDO_RECIPE_DIR

EXTRA_ARGS=("$@")

# ── Resolve serving endpoints ───────────────────────────────────────────────
resolve_serving_node() {
  local prefix="$1"
  squeue --user="${USER}" --noheader --format="%j %N" \
    | grep "^${prefix}" | head -1 | awk '{print $2}'
}

# Every metric here is a student solve rate, so the student is always required.
STUDENT_MODEL_NAME="${STUDENT_MODEL_NAME:-Llama-3.1-8B-Instruct-${USER}}"
STUDENT_NODE=$(resolve_serving_node "serve_${STUDENT_MODEL_NAME}")
if [[ -z "${STUDENT_NODE}" ]]; then
  echo "[transfer_eval] ERROR: no running SLURM job matching 'serve_${STUDENT_MODEL_NAME}*'. Launch the student server first." >&2
  exit 1
fi
export STUDENT_API_BASE="http://${STUDENT_NODE}:8080/v1"
export STUDENT_MODEL_NAME
echo "[transfer_eval] Student endpoint: ${STUDENT_API_BASE} (node=${STUDENT_NODE})"

if [[ " ${EXTRA_ARGS[*]:-} " == *" --gemini "* ]]; then
  : "${GEMINI_API_KEY:?--gemini requires GEMINI_API_KEY before sbatch}"
  echo "[transfer_eval] --gemini: leak/teaching judges via the Gemini API — local judge server not used"
else
  JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Qwen3.6-27B-${USER}}"
  JUDGE_NODE=$(resolve_serving_node "serve_${JUDGE_MODEL_NAME}")
  if [[ -z "${JUDGE_NODE}" ]]; then
    echo "[transfer_eval] ERROR: no running SLURM job matching 'serve_${JUDGE_MODEL_NAME}*'." >&2
    echo "[transfer_eval]        Launch it with eduardo/scripts/launch_judge_server.sh, or pass --gemini." >&2
    exit 1
  fi
  export JUDGE_API_BASE="http://${JUDGE_NODE}:8080/v1"
  export JUDGE_MODEL_NAME
  echo "[transfer_eval] Judge endpoint:   ${JUDGE_API_BASE} (node=${JUDGE_NODE})"
fi

# The 1-5 pedagogy/correctness judge (val-aux/* in wandb) is Gemini-only.
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "[transfer_eval] WARNING: GEMINI_API_KEY unset — the 1-5 eval judge will be skipped."
fi

if [[ ! -f "${TEST_PARQUET}" ]]; then
  echo "[transfer_eval] ERROR: eval parquet not found: ${TEST_PARQUET}" >&2
  echo "[transfer_eval]        Build it with \`python -m eduardo.process_dataset\` or set TEST_PARQUET." >&2
  exit 1
fi

echo "[transfer_eval] Evaluating model: ${EVAL_MODEL}"
echo "[transfer_eval] test.parquet:     ${TEST_PARQUET}"

# ── Caches on scratch, never in $HOME ───────────────────────────────────────
export XDG_CACHE_HOME="${WORK_DIR}/.cache"
export HF_HOME="${WORK_DIR}/huggingface"
export VLLM_CACHE_ROOT="${XDG_CACHE_HOME}/vllm"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME}/inductor"
mkdir -p "${HF_HOME}" "${VLLM_CACHE_ROOT}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}" \
         "${TORCHINDUCTOR_CACHE_DIR}"

# ── GH200 environment ─────────────────────────────────────────────────
export PYTHONPATH="${EDUARDO_RECIPE_DIR}/..:${PYTHONPATH:-}"
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES
unset VLLM_ATTENTION_BACKEND
ulimit -c 0

mkdir -p "${WORK_DIR}/eval_results"

# ── Container (pyxis) ───────────────────────────────────────────────────────
# Image built from the repo's Dockerfile. Set CONTAINER_ARGS="" to run on the host.
: "${CONTAINER_IMAGE:=${WORK_DIR}/eduardo.sqsh}"
: "${CONTAINER_MOUNTS:=${HOME}:${HOME},${WORK_DIR}:${WORK_DIR}}"
: "${CONTAINER_ARGS=--container-image=${CONTAINER_IMAGE} --container-mounts=${CONTAINER_MOUNTS}}"

srun --nodes=1 \
     --ntasks-per-node=1 \
     --export=ALL \
     ${CONTAINER_ARGS} \
     bash -lc "cd ${EDUARDO_RECIPE_DIR}/.. && python -m eval.run_env_test_eval \
       --model ${EVAL_MODEL} \
       --test-parquet ${TEST_PARQUET} \
       --tensor-parallel-size 4 \
       --output-dir ${WORK_DIR}/eval_results \
       ${EXTRA_ARGS[*]:-}"
