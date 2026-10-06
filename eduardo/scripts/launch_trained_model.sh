#!/usr/bin/env bash
# Serve a trained tutor (or a base model) with thinking enabled as a persistent
# vLLM endpoint (serve_<SERVED_NAME>, port 8080), e.g. for external benchmarks.
#
# Usage:
#   MODEL_PATH=${WORK_DIR}/eval_checkpoints/<experiment>/step_<step> \
#   SERVED_NAME=eduardo-27B ./eduardo/scripts/launch_trained_model.sh
#   MODEL_PATH=Qwen/Qwen3.8-27B SERVED_NAME=Qwen3.8-27B-thinking \
#       ./eduardo/scripts/launch_trained_model.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/serve_common.sh"

: "${MODEL_PATH:?set MODEL_PATH=<merged checkpoint dir or HF id>}"
SERVED_NAME="${SERVED_NAME:-eduardo-27B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-2}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

echo "[launch_trained_model] served-model-name=${SERVED_NAME}"
echo "[launch_trained_model] model=${MODEL_PATH}"
echo "[launch_trained_model] max-model-len=${MAX_MODEL_LEN}"
echo "[launch_trained_model] data-parallel-size=${DATA_PARALLEL_SIZE}"
echo "[launch_trained_model] tensor-parallel-size=${TENSOR_PARALLEL_SIZE}"

submit_vllm_server "${SERVED_NAME}" "${MODEL_PATH}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size "${DATA_PARALLEL_SIZE}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --default-chat-template-kwargs.enable_thinking true \
  --reasoning-parser qwen3 \
  --enable-chunked-prefill \
  --enable-prefix-caching
