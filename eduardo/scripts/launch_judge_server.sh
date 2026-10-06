#!/usr/bin/env bash
# Launch the judge (Qwen3.6-27B) as a persistent vLLM serving job
# (serve_<SERVED_NAME>, port 8080) used for the majority-vote reward gates.
# Layout on 1 node (4× GH200): 2 data-parallel engines × TP=2.
#
# Usage:
#   ./eduardo/scripts/launch_judge_server.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/serve_common.sh"

JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.6-27B}"
SERVED_NAME="${SERVED_NAME:-Qwen3.6-27B-$(whoami)}"
# Full transcripts rendered into the judge prompt can exceed 8K tokens.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-2}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
# Per-engine concurrent-sequence cap (512 slots total); excess requests queue
# in vLLM. Client pressure is bounded by reward_kwargs.judge_concurrency.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

echo "[launch_judge_server] served-model-name=${SERVED_NAME}"
echo "[launch_judge_server] model=${JUDGE_MODEL}"
echo "[launch_judge_server] max-model-len=${MAX_MODEL_LEN}"
echo "[launch_judge_server] data-parallel-size=${DATA_PARALLEL_SIZE}"
echo "[launch_judge_server] tensor-parallel-size=${TENSOR_PARALLEL_SIZE}"
echo "[launch_judge_server] max-num-seqs-per-engine=${MAX_NUM_SEQS}"

submit_vllm_server "${SERVED_NAME}" "${JUDGE_MODEL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size "${DATA_PARALLEL_SIZE}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --enable-chunked-prefill \
  --enable-prefix-caching
