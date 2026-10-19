#!/usr/bin/env bash
# Restart the physical baseline eval, hard expert collection, DAgger, and
# seed-11 MAPPO+/A6 training after a crash. Completed levels, teacher
# episodes, DAgger progress, and latest.pt checkpoints are reused.
set -u
ROOT=/srv/data/projects/kyy/xieton/repro
LOG_DIR="${HOME}/runs/isaac_dual_logs"
DATA_ROOT="${HOME}/runs/isaac_dual_data"
RUN_ROOT="${HOME}/runs"
BASE="${HOME}/runs/isaac_physical_baselines_seed11"
HARD_CFG="${ROOT}/configs/pursuit_v2/radar5_dual_hard.json"
LOG="${LOG_DIR}/physical_pipeline.log"
mkdir -p "$LOG_DIR" "$DATA_ROOT"
exec 9>"${LOG_DIR}/physical_pipeline.lock"
if ! flock -n 9; then
  echo "[pipeline] another maintainer already holds the lock"
  exit 0
fi
exec >>"$LOG" 2>&1

run_isaac() {
  cd "$ROOT"
  unset PYTHONPATH LD_LIBRARY_PATH CARB_APP_PATH EXP_PATH ISAAC_PATH ISAACSIM_PATH PYTHONSTARTUP || true
  # shellcheck disable=SC1091
  source /opt/lab/apps/miniforge3-26.7.2/etc/profile.d/conda.sh
  conda activate "${HOME}/.conda/envs/omnidrones-isaac41-writable"
  export ISAACSIM_PATH=/opt/lab/apps/isaac-sim-4.1.0
  export OMNIDRONES_SOURCE=/opt/lab/src/OmniDrones-isaac41
  set +u
  # shellcheck disable=SC1091
  source "${ISAACSIM_PATH}/setup_conda_env.sh"
  set -u
  export PYTHONPATH="/srv/data/projects/kyy/omnidrones:${ROOT}:${PYTHONPATH:-}"
  export PYTHONUNBUFFERED=1
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
  export OMNI_USER_CACHE="${OMNI_USER_CACHE:-${HOME}/.cache/isaac-kit}"
  unset DISPLAY || true
  export OMNI_KIT_ACCEPT_EULA=YES
  "$@"
}

summaries_ready() {
  local method summary
  for method in mpc frpn apf; do
    summary="${BASE}/isaac_dual_eval_${method}/summary.json"
    [[ -f "$summary" ]] || return 1
    python3 - "$summary" <<'PY' || return 1
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
levels = [row.get("level") for row in payload.get("summaries") or []]
raise SystemExit(0 if levels == ["nominal", "medium", "hard", "severe", "extreme"] else 1)
PY
  done
}

prior_ready() {
  [[ -f "${DATA_ROOT}/hard_mpc_prior.pt" && -f "${DATA_ROOT}/hard_mpc_prior.json" ]] || return 1
  python3 - "${DATA_ROOT}/hard_mpc_prior.json" <<'PY' || return 1
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if int(payload.get("n_episodes", 0)) >= 200 else 1)
PY
}

dagger_ready() {
  [[ -f "${DATA_ROOT}/dagger_hard_seed11/best_imitation.pt" ]] \
    && [[ -f "${DATA_ROOT}/dagger_hard_seed11/expert_episodes.pt" ]] \
    && [[ -f "${DATA_ROOT}/dagger_hard_seed11/correction_episodes.pt" ]] \
    && [[ -f "${DATA_ROOT}/dagger_hard_seed11/imitation_manifest.json" ]]
}

echo "[pipeline] $(date) maintainer started"

failures=0
while true; do
  if summaries_ready; then
    echo "[pipeline] $(date) baselines complete"
  else
    echo "[pipeline] $(date) baseline eval"
    if run_isaac python evaluate_isaac_baselines.py \
      --methods mpc frpn apf \
      --levels nominal medium hard severe extreme \
      --seed-start 7000000 --seed-count 200 \
      --log-every 25 \
      --out-root "$BASE"; then
      failures=0
    else
      failures=$((failures + 1))
      echo "[pipeline] $(date) baseline eval exited, retry ${failures}"
    fi
    sleep $((failures > 0 ? (failures > 5 ? 300 : 30) : 5))
    continue
  fi

  if prior_ready; then
    echo "[pipeline] $(date) hard MPC prior complete"
  else
    echo "[pipeline] $(date) hard MPC prior"
    if run_isaac python collect_isaac_pursuit_data.py \
      --env-config "$HARD_CFG" \
      --teacher mpc --demo-seed 2000000 --rollouts 200 \
      --dataset "${DATA_ROOT}/hard_mpc_prior.pt"; then
      failures=0
    else
      failures=$((failures + 1))
      echo "[pipeline] $(date) MPC prior exited, retry ${failures}"
    fi
    sleep $((failures > 0 ? (failures > 5 ? 300 : 30) : 5))
    continue
  fi

  if dagger_ready; then
    echo "[pipeline] $(date) DAgger complete"
  else
    echo "[pipeline] $(date) hard DAgger seed=11"
    if run_isaac python pretrain_isaac_pursuit_actor.py \
      --env-config "$HARD_CFG" \
      --seed 11 --demo-seed 2000000 \
      --bc-episodes 50 --bc-updates 1500 \
      --dagger-rounds 2 --dagger-episodes 40 --dagger-updates 800 \
      --out "${DATA_ROOT}/dagger_hard_seed11"; then
      failures=0
    else
      failures=$((failures + 1))
      echo "[pipeline] $(date) DAgger exited, retry ${failures}"
    fi
    sleep $((failures > 0 ? (failures > 5 ? 300 : 30) : 5))
    continue
  fi

  mappo_out="${RUN_ROOT}/isaac_dual_mappo_plus_hard_seed11"
  if [[ -f "${mappo_out}/final.pt" ]]; then
    echo "[pipeline] $(date) MAPPO+ complete"
  else
    echo "[pipeline] $(date) MAPPO+ seed=11"
    resume=()
    if [[ -f "${mappo_out}/latest.pt" ]]; then
      resume=(--checkpoint "${mappo_out}/latest.pt")
      echo "[pipeline] resume MAPPO from ${mappo_out}/latest.pt"
    fi
    if run_isaac python train_isaac_pursuit.py mappo --variant mappo_plus \
      --env-config "$HARD_CFG" \
      --actor-init "${DATA_ROOT}/dagger_hard_seed11/best_imitation.pt" \
      --demos "${DATA_ROOT}/dagger_hard_seed11/expert_episodes.pt" \
      --aux-demos "${DATA_ROOT}/dagger_hard_seed11/correction_episodes.pt" \
      --episodes 1000 --seed 11 \
      --budget-checkpoints 100,300,1000 \
      --batch-size 1024 --hidden-dim 128 \
      --out "$mappo_out" \
      "${resume[@]}"; then
      failures=0
    else
      failures=$((failures + 1))
      echo "[pipeline] $(date) MAPPO+ exited, retry ${failures}"
    fi
    sleep $((failures > 0 ? (failures > 5 ? 300 : 30) : 5))
    continue
  fi

  # Pure-actor A6. Do not resume isaac_dual_A6_hard_seed11: that run executed an MPC blend.
  a6_out="${RUN_ROOT}/isaac_dual_A6_pure_hard_seed11"
  if [[ -f "${a6_out}/final.pt" ]]; then
    echo "[pipeline] $(date) A6 complete"
    echo "[pipeline] $(date) DONE"
    exit 0
  fi
  echo "[pipeline] $(date) A6 seed=11"
  resume=()
  if [[ -f "${a6_out}/latest.pt" ]]; then
    resume=(--checkpoint "${a6_out}/latest.pt")
    echo "[pipeline] resume A6 from ${a6_out}/latest.pt"
  fi
  if run_isaac python train_isaac_pursuit.py matd3 --variant A6 \
    --env-config "$HARD_CFG" \
    --actor-init "${DATA_ROOT}/dagger_hard_seed11/best_imitation.pt" \
    --prior "${DATA_ROOT}/hard_mpc_prior.pt" \
    --episodes 1000 --seed 11 \
    --budget-checkpoints 100,300,1000 \
    --batch-size 256 --hidden-dim 128 \
    --out "$a6_out" \
    "${resume[@]}"; then
    failures=0
  else
    failures=$((failures + 1))
    echo "[pipeline] $(date) A6 exited, retry ${failures}"
    sleep $((failures > 5 ? 300 : 30))
    continue
  fi
done
