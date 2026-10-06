#!/bin/bash
#SBATCH --job-name=eduardo-eval
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
# Eduardo evaluation. MathDial / Eedi with --gemini need no servers; other
# benchmarks need the student server (launch_student_server.sh) and, unless
# --gemini or EVAL_NO_JUDGE=1 is set, the judge server (launch_judge_server.sh).
#
# Env vars:
#   SERVING_API_KEY     bearer token of the vLLM serving endpoints
#   EVAL_MODEL          HF hub id or local path of the tutor model
#   WORK_DIR            scratch dir for results and caches
#   EDUARDO_RECIPE_DIR  path to eduardo/ in the checkout
#   EVAL_NO_JUDGE       optional: 1 to skip LLM-judge scoring
#   GEMINI_API_KEY      required with --gemini (no local judge server needed)
#
# Usage:
#   sbatch --export=ALL,EVAL_MODEL=eth-nlped/TutorRL-7B eval/eval_slurm.sh
#   sbatch --export=ALL,EVAL_MODEL=/path/to/exported/step_375 eval/eval_slurm.sh \
#       --num-problems 100 --max-teacher-turns 8
#   sbatch --export=ALL,EVAL_MODEL=/path/to/eduardo-27B eval/eval_slurm.sh \
#       --gemini --benchmark eedi

set -euo pipefail

# ── Required env ────────────────────────────────────────────────────────────
: "${SERVING_API_KEY:?set SERVING_API_KEY before sbatch}"
: "${EVAL_MODEL:?set EVAL_MODEL=<hf-id-or-local-path> before sbatch}"
: "${WORK_DIR:=${SCRATCH:-$HOME/scratch}/eduardo}"
: "${EDUARDO_RECIPE_DIR:=$HOME/eduardo/eduardo}"
export WORK_DIR EDUARDO_RECIPE_DIR

EXTRA_ARGS=("$@")

# ── Resolve serving endpoints ───────────────────────────────────────────────
resolve_serving_node() {
  local prefix="$1"
  squeue --user="${USER}" --noheader --format="%j %N" \
    | grep "^${prefix}" | head -1 | awk '{print $2}'
}

# The mathdial/eedi continuation benchmarks do not use the student simulator.
NEEDS_STUDENT=1
case " ${EXTRA_ARGS[*]:-} " in
  *" mathdial "*|*" eedi "*) NEEDS_STUDENT=0 ;;
esac

if [[ "${NEEDS_STUDENT}" == "1" ]]; then
  STUDENT_MODEL_NAME="${STUDENT_MODEL_NAME:-Llama-3.1-8B-Instruct-${USER}}"
  STUDENT_NODE=$(resolve_serving_node "serve_${STUDENT_MODEL_NAME}")
  if [[ -z "${STUDENT_NODE}" ]]; then
    echo "[eval_slurm] ERROR: no running SLURM job matching 'serve_${STUDENT_MODEL_NAME}*'. Launch the student server first." >&2
    exit 1
  fi
  export STUDENT_API_BASE="http://${STUDENT_NODE}:8080/v1"
  export STUDENT_MODEL_NAME
  echo "[eval_slurm] Student endpoint: ${STUDENT_API_BASE} (node=${STUDENT_NODE})"
else
  echo "[eval_slurm] Student server not required for this benchmark/stage"
fi

JUDGE_ARGS=""
if [[ "${EVAL_NO_JUDGE:-0}" == "1" ]]; then
  JUDGE_ARGS="--no-judge"
  echo "[eval_slurm] EVAL_NO_JUDGE=1 — skipping LLM-judge scoring"
elif [[ " ${EXTRA_ARGS[*]:-} " == *" --gemini "* ]]; then
  : "${GEMINI_API_KEY:?--gemini requires GEMINI_API_KEY before sbatch}"
  echo "[eval_slurm] --gemini: judging via the Gemini API — local judge server not used"
else
  JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Qwen3.6-27B-${USER}}"
  JUDGE_NODE=$(resolve_serving_node "serve_${JUDGE_MODEL_NAME}")
  if [[ -z "${JUDGE_NODE}" ]]; then
    echo "[eval_slurm] ERROR: no running SLURM job matching 'serve_${JUDGE_MODEL_NAME}*'." >&2
    echo "[eval_slurm]        Launch it with eduardo/scripts/launch_judge_server.sh, or set EVAL_NO_JUDGE=1 to skip judge scoring." >&2
    exit 1
  fi
  export JUDGE_API_BASE="http://${JUDGE_NODE}:8080/v1"
  export JUDGE_MODEL_NAME
  echo "[eval_slurm] Judge endpoint:   ${JUDGE_API_BASE} (node=${JUDGE_NODE})"
fi

echo "[eval_slurm] Evaluating model: ${EVAL_MODEL}"

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
     bash -lc "cd ${EDUARDO_RECIPE_DIR}/.. && python -m eval.run_middleturn_eval \
       --model ${EVAL_MODEL} \
       --tensor-parallel-size 4 \
       --output-dir ${WORK_DIR}/eval_results \
       ${JUDGE_ARGS} ${EXTRA_ARGS[*]:-}"
