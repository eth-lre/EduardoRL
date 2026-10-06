#!/usr/bin/env bash
# Launch the student simulator (Llama-3.1-8B-Instruct) as a persistent vLLM
# serving job (serve_<SERVED_NAME>, port 8080) that outlives training jobs.
# Layout on 1 node (4× GH200): 4 data-parallel engines × TP=1, which gives more
# KV cache and throughput than TP=4 for an 8B model.
#
# Usage:
#   ./eduardo/scripts/launch_student_server.sh
#   STUDENT_MODEL=/local/path/Llama-3.1-8B-Instruct ./eduardo/scripts/launch_student_server.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/serve_common.sh"

STUDENT_MODEL="${STUDENT_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
SERVED_NAME="${SERVED_NAME:-Llama-3.1-8B-Instruct-$(whoami)}"
# Student requests include the full conversation, so length scales with
# max_user_turns × max_tokens_per_turn.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-4}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
# 4 engines × 256 = 1024 slots, matching the client-side budget in
# configs/user.yaml (8 reward workers × student_concurrency=128).
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

echo "[launch_student_server] served-model-name=${SERVED_NAME}"
echo "[launch_student_server] model=${STUDENT_MODEL}"
echo "[launch_student_server] max-model-len=${MAX_MODEL_LEN}"
echo "[launch_student_server] data-parallel-size=${DATA_PARALLEL_SIZE}"
echo "[launch_student_server] tensor-parallel-size=${TENSOR_PARALLEL_SIZE}"
echo "[launch_student_server] max-num-seqs-per-engine=${MAX_NUM_SEQS}"

submit_vllm_server "${SERVED_NAME}" "${STUDENT_MODEL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size "${DATA_PARALLEL_SIZE}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --enable-chunked-prefill \
  --enable-prefix-caching
