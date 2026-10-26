#!/usr/bin/env bash
# Finish remaining postlc multi-level held-out evals with parallel Isaac envs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
RUNS="/srv/data/projects/kyy/runs/isaac_dual_postlc"
LOG_DIR="${RUNS}/logs"
mkdir -p "${LOG_DIR}"
NUM_ENVS="${NUM_ENVS:-8}"
SEED_START="${SEED_START:-7000000}"
SEED_COUNT="${SEED_COUNT:-100}"

source /opt/lab/apps/miniforge3-26.7.2/etc/profile.d/conda.sh
conda activate /home/wb/.conda/envs/omnidrones-isaac41-writable
export ISAACSIM_PATH=/opt/lab/apps/isaac-sim-4.1.0
set +u; source "${ISAACSIM_PATH}/setup_conda_env.sh"; set -u
export PYTHONPATH="/srv/data/projects/kyy/omnidrones:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${HOME}/.cache"
export OMNI_USER_CACHE="${HOME}/.cache/isaac-kit"
unset DISPLAY || true
cd "${ROOT}"

run_one() {
  local method="$1" out="$2" ckpt="${3:-}"
  local levels=(nominal medium hard severe extreme)
  local args=(
    -u "${ROOT}/evaluate_isaac_pursuit.py"
    --method "${method}"
    --levels "${levels[@]}"
    --skip-existing-levels
    --seed-start "${SEED_START}"
    --seed-count "${SEED_COUNT}"
    --num-envs "${NUM_ENVS}"
    --log-every 10
    --out "${out}"
  )
  if [[ -n "${ckpt}" ]]; then
    args+=(--checkpoint "${ckpt}")
  fi
  echo "[queue] $(date -Is) method=${method} num_envs=${NUM_ENVS} out=${out}"
  python "${args[@]}"
}

echo "[queue] start $(date -Is) NUM_ENVS=${NUM_ENVS}"

run_one apf "${RUNS}/train/isaac_dual_eval_baseline_apf_hardonly_postlc"
run_one frpn "${RUNS}/train/isaac_dual_eval_baseline_frpn_hardonly_postlc"
run_one mpc "${RUNS}/train/isaac_dual_eval_baseline_mpc_hardonly_postlc"
run_one mappo "${RUNS}/train/isaac_dual_eval_mappo_mpc_hardonly_postlc_seed11" \
  "${RUNS}/train/isaac_dual_mappo_plus_mpc_hard_postlc_seed11/best_capture.pt"
run_one mappo "${RUNS}/train/isaac_dual_eval_mappo_frpn_hardonly_postlc_seed11" \
  "${RUNS}/train/isaac_dual_mappo_plus_frpn_hard_postlc_seed11/best_capture.pt"
run_one matd3 "${RUNS}/train/isaac_dual_eval_matd3_mpc_hardonly_postlc_seed11" \
  "${RUNS}/train/isaac_dual_A6_mpc_hard_postlc_seed11/best_capture.pt"
run_one matd3 "${RUNS}/train/isaac_dual_eval_matd3_frpn_hardonly_postlc_seed11" \
  "${RUNS}/train/isaac_dual_A6_frpn_hard_postlc_seed11/best_capture.pt"

echo "[queue] done $(date -Is)"
