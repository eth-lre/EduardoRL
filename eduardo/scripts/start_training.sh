#!/bin/bash
# Inner launcher run on each allocated node by train_slurm.sh: sets up the Ray
# cluster and launches verl.trainer.main_ppo on the head node.
#
# Args: <config_name> [extra Hydra overrides …]
set -euo pipefail

CONFIG_NAME="${1:-eduardo_base}"
shift || true

: "${EDUARDO_RECIPE_DIR:?must be set by train_slurm.sh}"
: "${WORK_DIR:?must be set by train_slurm.sh}"
: "${EXPERIMENT:?must be set by train_slurm.sh}"

echo "[start_training] CONFIG_NAME=${CONFIG_NAME}"
echo "[start_training] EDUARDO_RECIPE_DIR=${EDUARDO_RECIPE_DIR}"
echo "[start_training] WORK_DIR=${WORK_DIR}"
echo "[start_training] EXPERIMENT=${EXPERIMENT}"
echo "[start_training] extra overrides: $*"

export PYTHONPATH="${EDUARDO_RECIPE_DIR}/..:${PYTHONPATH:-}"

# ── Ray cluster setup ───────────────────────────────────────────────────────
ray stop -f || true
NNODES="${SLURM_NNODES:-1}"
NODE_ID="${SLURM_NODEID:-0}"

echo "[start_training] NNODES=${NNODES}, NODE_ID=${NODE_ID}"

if [[ "${NNODES}" -eq 1 ]]; then
    # Single node: Ray head only.
    echo "[start_training] Single-node mode: starting Ray head..."
    ray start --head \
              --num-gpus="${SLURM_GPUS_ON_NODE:-4}" \
              --temp-dir="${RAY_TMPDIR}" \
              --include-dashboard=false
else
    # Multi-node: head + workers.
    HEAD_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
    HEAD_IP=$(getent hosts "${HEAD_NODE}" | awk '{print $1}')
    RAY_PORT=6379

    echo "[start_training] Multi-node mode: HEAD_NODE=${HEAD_NODE}, HEAD_IP=${HEAD_IP}"

    if [[ "${NODE_ID}" == "0" ]]; then
        echo "[start_training] Starting Ray head on node 0..."
        ray start --head \
                  --node-ip-address="${HEAD_IP}" \
                  --port="${RAY_PORT}" \
                  --num-gpus="${SLURM_GPUS_ON_NODE:-4}" \
                  --temp-dir="${RAY_TMPDIR}" \
                  --include-dashboard=false
        
        echo "[start_training] Waiting for ${NNODES} nodes to join Ray cluster..."
        NODES_CONNECTED=false
        for i in {1..60}; do
            NODE_COUNT=$(python -c "import ray; ray.init(address='auto', ignore_reinit_error=True); print(len([n for n in ray.nodes() if n['Alive']]))" 2>/dev/null || echo "0")
            NODE_COUNT="${NODE_COUNT:-0}"
            if [[ "${NODE_COUNT}" -ge "${NNODES}" ]]; then
                echo "[start_training] All ${NNODES} nodes connected!"
                NODES_CONNECTED=true
                break
            fi
            echo "[start_training] Waiting for nodes... (${NODE_COUNT}/${NNODES}, attempt ${i}/60)"
            sleep 5
        done
        if [[ "${NODES_CONNECTED}" != "true" ]]; then
            echo "[start_training] ERROR: Only ${NODE_COUNT}/${NNODES} nodes connected after 5 minutes. Aborting." >&2
            exit 1
        fi
    else
        echo "[start_training] Starting Ray worker on node ${NODE_ID}, connecting to ${HEAD_IP}:${RAY_PORT}..."
        sleep 10
        ray start --address="${HEAD_IP}:${RAY_PORT}" \
                  --num-gpus="${SLURM_GPUS_ON_NODE:-4}" \
                  --temp-dir="${RAY_TMPDIR}"
        
        # Workers stay alive until the head's Ray cluster shuts down.
        echo "[start_training] Worker node ${NODE_ID} waiting for training to complete..."
        while true; do
            sleep 60
            if ! ray status &>/dev/null; then
                echo "[start_training] Ray stopped on worker node ${NODE_ID}, exiting."
                exit 0
            fi
        done
    fi
fi

# ── Build dataset and run training (head node only) ─────────────────────────
TRAIN_PARQUET="${WORK_DIR}/data/eduardo/train.parquet"
EXPECTED_AGENT_NAME="eduardo_agent"
need_rebuild=1
if [[ -f "${TRAIN_PARQUET}" ]]; then
  actual_agent_name=$(python -c "import pandas as pd, sys; df = pd.read_parquet('${TRAIN_PARQUET}', columns=['agent_name']); print(df['agent_name'].iloc[0] if len(df) else '')" 2>/dev/null || echo "")
  if [[ "${actual_agent_name}" == "${EXPECTED_AGENT_NAME}" ]]; then
    need_rebuild=0
  else
    echo "[start_training] parquet agent_name=${actual_agent_name:-<missing>} != ${EXPECTED_AGENT_NAME}; rebuilding dataset"
  fi
fi

if [[ ${need_rebuild} -eq 1 ]]; then
  echo "[start_training] building dataset…"
  python -m eduardo.process_dataset \
    --num-train 10000 --num-test 100 --seed 42 \
    --out-dir "${WORK_DIR}/data/eduardo"
fi

# Derive resource settings from the SLURM allocation.
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))

echo "[start_training] Auto-configuring: nnodes=${NNODES}, gpus_per_node=${GPUS_PER_NODE}, fsdp_size=${TOTAL_GPUS}"

export RAY_ADDRESS="auto"

python -m verl.trainer.main_ppo \
  --config-path "${EDUARDO_RECIPE_DIR}/configs" \
  --config-name "${CONFIG_NAME}" \
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="${TOTAL_GPUS}" \
  "$@"
