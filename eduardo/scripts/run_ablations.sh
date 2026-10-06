#!/bin/bash
# Ablation launcher: three single-factor arms relative to the baseline in
# configs/user.yaml, each applied as Hydra overrides to train_slurm.sh.
#
# Arms:
#   nomasking   omit_teacher_turns=false — the graded final attempt sees the
#               teacher's actual turns instead of "(hidden)".
#   nogates     judge_hard_gate=false + judge_penalty=0 — no pedagogy penalty;
#               judges still run and log judge/leak_rate.
#   notransfer  test_on_transfer=false — final attempt and Δ_solve baseline use
#               the original problem rather than the near-transfer variant.
#
# Usage:
#   export WANDB_ENTITY=<team> WANDB_API_KEY=… SERVING_API_KEY=<token>
#   export MODEL_PATH=Qwen/Qwen3.5-4B
#   ./scripts/run_ablations.sh [--dry-run]

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  echo "[ablations] dry-run mode — commands will be printed, not submitted."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_slurm.sh"
CONFIG_FILE="${SCRIPT_DIR}/../configs/user.yaml"
CONFIG_NAME="eduardo_base"
BASE_JOB_NAME="eduardo"

# Shared W&B group so the arms appear together.
GROUP_NAME="${GROUP_NAME:-eduardo-ablations-4b}"

RK="custom_reward_function.reward_kwargs"

# Applied to every arm. The 4B policy fits on one GPU, so rollout TP=1 avoids
# cross-rank vLLM worker synchronization.
COMMON_OVERRIDES="actor_rollout_ref.rollout.tensor_model_parallel_size=1"

# ── Submit-time env (train_slurm.sh re-checks these at job start) ───────────
: "${SERVING_API_KEY:?set SERVING_API_KEY before running}"
: "${WANDB_API_KEY:?set WANDB_API_KEY before running}"
: "${WANDB_ENTITY:?set WANDB_ENTITY before running (team name)}"

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "[ablations] WARNING: MODEL_PATH unset — all arms will use the default"
  echo "[ablations]          in configs/user.yaml, not a 4B model."
fi

# ── Pre-flight: check the baseline values each arm flips ────────────────────
assert_baseline() {
  local key="$1" want="$2" actual
  actual=$(grep -E "^[[:space:]]*${key}:" "${CONFIG_FILE}" \
           | head -1 | sed -E 's/^[[:space:]]*[^:]+:[[:space:]]*([^[:space:]#]+).*/\1/')
  if [[ "${actual}" != "${want}" ]]; then
    echo "[ablations] WARNING: user.yaml has ${key}=${actual:-<unset>}, expected"
    echo "[ablations]          baseline ${want}. That arm may be a no-op."
  fi
}
assert_baseline "omit_teacher_turns" "true"
assert_baseline "judge_hard_gate"    "true"
assert_baseline "test_on_transfer"   "true"

# ── Per-job submission helper ──────────────────────────────────────────────
submit_job() {
  local exp_name="$1"
  local overrides="$2"

  local export_vars="ALL,EXPERIMENT=${exp_name}"
  [[ -n "${MODEL_PATH:-}" ]] && export_vars+=",MODEL_PATH=${MODEL_PATH}"

  echo "----------------------------------------------------------------"
  echo "[ablations] ${exp_name}"
  echo "            overrides: ${COMMON_OVERRIDES} ${overrides}"

  local sbatch_cmd=(
    sbatch
    --job-name="${BASE_JOB_NAME}-${exp_name}"
    --export="${export_vars}"
    "${TRAIN_SCRIPT}"
    "${CONFIG_NAME}"
    ${COMMON_OVERRIDES}
    ${overrides}
    "trainer.group_name=${GROUP_NAME}"
  )

  if ${DRY_RUN}; then
    echo "            ${sbatch_cmd[*]}"
  else
    "${sbatch_cmd[@]}"
  fi
}

# ── Arms ────────────────────────────────────────────────────────────────────
# No teacher-turn masking.
submit_job "ablation-4b-nomasking" \
  "${RK}.omit_teacher_turns=false"

# No pedagogy penalty. judge_penalty=0 is needed because disabling the hard
# gate otherwise activates the additive r_judge = -judge_penalty term.
submit_job "ablation-4b-nogates" \
  "${RK}.judge_hard_gate=false ${RK}.judge_penalty=0"

# No near-transfer testing: Δ_solve against pre_solve_rate on the original problem.
submit_job "ablation-4b-notransfer" \
  "${RK}.test_on_transfer=false"

echo "----------------------------------------------------------------"
if ${DRY_RUN}; then
  echo "[ablations] dry run complete — nothing submitted."
else
  echo "[ablations] 3 jobs submitted. Verify each arm from its W&B config:"
  echo "[ablations]   nomasking  -> reward_kwargs.omit_teacher_turns == false"
  echo "[ablations]   nogates    -> reward/gated_rate == 0, reward/judge_penalty == 0"
  echo "[ablations]   notransfer -> accuracy/transfer_tested_rate == 0"
fi
