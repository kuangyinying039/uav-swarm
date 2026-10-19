#!/usr/bin/env bash
# Isaac PhysX dual-capture suite (sample-efficiency protocol).
# Protocol: isaac_dual_experiment_manifest.json / ISAAC_DUAL_EXPERIMENTS.md
#
# Default training budget is 1000 episodes with snapshots at 250/500/750/1000.
# Weak-communication experiments are intentionally out of scope here.
#
# Usage:
#   bash run_isaac_dual_suite.sh env
#   bash run_isaac_dual_suite.sh data
#   bash run_isaac_dual_suite.sh dagger
#   bash run_isaac_dual_suite.sh train-compare
#   bash run_isaac_dual_suite.sh train-ablation-key
#   bash run_isaac_dual_suite.sh train-reward-ablation   # seed 11 only by default
#   bash run_isaac_dual_suite.sh eval-compare
#   bash run_isaac_dual_suite.sh eval-budgets
#   bash run_isaac_dual_suite.sh eval-ablation-key
#   bash run_isaac_dual_suite.sh extras
#   bash run_isaac_dual_suite.sh all
#
# Overrides:
#   EPISODES=1000 BUDGET_CHECKPOINTS=250,500,750,1000 SEEDS='11 12 13'
#   FAST_EVAL=1              # medium × 50 for quick iteration (~8–10× faster eval)
#   EVAL_SEED_COUNT=200 EVAL_LEVELS='nominal medium hard extreme'
#   SKIP_CONT_EVAL=1         # skip A5 continue formal eval (default)
#   EVAL_FORCE=1             # re-run even if summary.json exists

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN_ROOT="${RUN_ROOT:-$HOME/runs}"
DATA_ROOT="${DATA_ROOT:-$RUN_ROOT/isaac_dual_data}"
SEEDS=(${SEEDS:-11 12 13})
EPISODES="${EPISODES:-1000}"
BUDGET_CHECKPOINTS="${BUDGET_CHECKPOINTS:-250,500,750,1000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
EVAL_SEED_START="${EVAL_SEED_START:-7000000}"
EVAL_SEED_COUNT="${EVAL_SEED_COUNT:-200}"
# Space-separated difficulty list for evaluate_isaac_pursuit.py --levels
EVAL_LEVELS="${EVAL_LEVELS:-nominal medium hard severe extreme}"
# Skip A5 seed13 continue formal eval (expensive; enable with SKIP_CONT_EVAL=0).
SKIP_CONT_EVAL="${SKIP_CONT_EVAL:-1}"
# FAST_EVAL=1 → medium-only, 50 episodes: ~8–10× faster for algorithm iteration.
# Paper tables should still use FAST_EVAL=0 (200 × four levels).
FAST_EVAL="${FAST_EVAL:-0}"
if [[ "${FAST_EVAL}" == "1" ]]; then
  EVAL_SEED_COUNT="${EVAL_SEED_COUNT_FAST:-50}"
  EVAL_LEVELS="${EVAL_LEVELS_FAST:-medium}"
  SKIP_CONT_EVAL=1
  echo "[fast-eval] EVAL_SEED_COUNT=${EVAL_SEED_COUNT} EVAL_LEVELS='${EVAL_LEVELS}'"
fi
BUDGET_EVAL_COUNT="${BUDGET_EVAL_COUNT:-50}"
EXTRAS_SEED_COUNT="${EXTRAS_SEED_COUNT:-50}"
MEDIUM_CFG="configs/pursuit_v2/radar5_dual_medium.json"
HARD_CFG="configs/pursuit_v2/radar5_dual_hard.json"

activate_env() {
  unset PYTHONPATH LD_LIBRARY_PATH CARB_APP_PATH EXP_PATH ISAAC_PATH ISAACSIM_PATH PYTHONSTARTUP || true
  # shellcheck disable=SC1091
  source /opt/lab/apps/miniforge3-26.7.2/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-$HOME/.conda/envs/omnidrones-isaac41-writable}"
  export ISAACSIM_PATH="${ISAACSIM_PATH:-/opt/lab/apps/isaac-sim-4.1.0}"
  export OMNIDRONES_SOURCE="${OMNIDRONES_SOURCE:-/opt/lab/src/OmniDrones-isaac41}"
  set +u
  # shellcheck disable=SC1091
  source "$ISAACSIM_PATH/setup_conda_env.sh"
  set -u
  cd "$ROOT"
  export PYTHONPATH="/srv/data/projects/kyy/omnidrones:$ROOT:${PYTHONPATH:-}"
  export PYTHONUNBUFFERED=1
  # Avoid PermissionError / exclusive-lock hangs on the shared Isaac kit cache.
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
  mkdir -p "$HOME/.cache/isaac-kit" "$HOME/.cache/ov"
  export OMNI_USER_CACHE="${OMNI_USER_CACHE:-$HOME/.cache/isaac-kit}"
  # Headless training must not inherit a dead DISPLAY; GLFW then fails and the
  # GUI kit path can SIGSEGV in SimulationApp._prepare_ui.
  unset DISPLAY || true
  export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
}

# Retry Isaac python entrypoints after Kit startup crashes (SIGSEGV/SIGABRT).
# Training logic errors (non-signal exits) are not retried.
run_isaac_python() {
  local max_attempts="${ISAAC_START_RETRIES:-3}"
  local attempt=1
  local rc=0
  while (( attempt <= max_attempts )); do
    echo "[isaac-run] attempt=${attempt}/${max_attempts}: $*"
    set +e
    "$@"
    rc=$?
    set -e
    if (( rc == 0 )); then
      return 0
    fi
    # 139=SIGSEGV, 134=SIGABRT, 132=SIGILL — typical Kit startup failures.
    if (( rc == 139 || rc == 134 || rc == 132 )); then
      echo "[isaac-run] Kit crashed (exit=${rc}); clearing user locks and retrying" >&2
      rm -f "${XDG_CACHE_HOME:-$HOME/.cache}/ov/_cache.lock" \
            "${OMNI_USER_CACHE:-$HOME/.cache/isaac-kit}/_cache.lock" 2>/dev/null || true
      sleep $(( attempt * 20 ))
      attempt=$((attempt + 1))
      continue
    fi
    return "$rc"
  done
  echo "[isaac-run] giving up after ${max_attempts} Kit crashes (last exit=${rc})" >&2
  return "$rc"
}

print_env() {
  cat <<'EOF'
# Paste into each tmux window before running jobs:
unset PYTHONPATH LD_LIBRARY_PATH CARB_APP_PATH EXP_PATH ISAAC_PATH ISAACSIM_PATH PYTHONSTARTUP
source /opt/lab/apps/miniforge3-26.7.2/etc/profile.d/conda.sh
conda activate "$HOME/.conda/envs/omnidrones-isaac41-writable"
export ISAACSIM_PATH=/opt/lab/apps/isaac-sim-4.1.0
export OMNIDRONES_SOURCE=/opt/lab/src/OmniDrones-isaac41
source "$ISAACSIM_PATH/setup_conda_env.sh"
cd /srv/data/projects/kyy/xieton/repro
export PYTHONPATH="/srv/data/projects/kyy/omnidrones:/srv/data/projects/kyy/xieton/repro:${PYTHONPATH}"
EOF
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing required artifact: $path" >&2
    exit 1
  fi
}

prior_complete() {
  local json_path="$1"
  local expected="$2"
  python - "$json_path" "$expected" <<'PY'
import json, sys
from pathlib import Path
path, expected = Path(sys.argv[1]), int(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text())
raise SystemExit(0 if int(payload.get("n_episodes", 0)) >= expected else 1)
PY
}

phase_data() {
  mkdir -p "$DATA_ROOT"
  if prior_complete "$DATA_ROOT/hard_mpc_prior.json" 200; then
    echo "[data] skip hard MPC prior (already 200 episodes)"
  else
    echo "[data] hard MPC prior → $DATA_ROOT/hard_mpc_prior.pt"
    run_isaac_python python collect_isaac_pursuit_data.py \
      --env-config "$HARD_CFG" \
      --teacher mpc --demo-seed 2000000 --rollouts 200 \
      --dataset "$DATA_ROOT/hard_mpc_prior.pt"
  fi
}

phase_dagger() {
  require_file "$DATA_ROOT/hard_mpc_prior.pt"
  for seed in "${SEEDS[@]}"; do
    if [[ -f "$DATA_ROOT/dagger_hard_seed${seed}/best_imitation.pt" ]]; then
      echo "[dagger] skip seed=$seed (best_imitation.pt exists)"
      continue
    fi
    echo "[dagger] hard seed=$seed"
    run_isaac_python python pretrain_isaac_pursuit_actor.py \
      --env-config "$HARD_CFG" \
      --seed "$seed" --demo-seed 2000000 \
      --bc-episodes 50 --bc-updates 1500 \
      --dagger-rounds 2 --dagger-episodes 40 --dagger-updates 800 \
      --out "$DATA_ROOT/dagger_hard_seed${seed}"
    require_file "$DATA_ROOT/dagger_hard_seed${seed}/best_imitation.pt"
  done
}

train_mappo_dagger() {
  local seed="$1"
  local out="$RUN_ROOT/isaac_dual_mappo_dagger_hard_seed${seed}"
  if [[ -f "$out/final.pt" ]]; then
    echo "[train] skip mappo_dagger seed=$seed (final.pt exists)"
    return 0
  fi
  require_file "$DATA_ROOT/dagger_hard_seed${seed}/best_imitation.pt"
  local resume=()
  if [[ -f "$out/latest.pt" ]]; then
    resume=(--checkpoint "$out/latest.pt")
    echo "[train] resume mappo_dagger seed=$seed from latest.pt"
  fi
  run_isaac_python python train_isaac_pursuit.py mappo --variant mappo_dagger \
    --env-config "$HARD_CFG" \
    --actor-init "$DATA_ROOT/dagger_hard_seed${seed}/best_imitation.pt" \
    --demos "$DATA_ROOT/dagger_hard_seed${seed}/expert_episodes.pt" \
    --aux-demos "$DATA_ROOT/dagger_hard_seed${seed}/correction_episodes.pt" \
    --episodes "$EPISODES" --seed "$seed" \
    --budget-checkpoints "$BUDGET_CHECKPOINTS" \
    --batch-size "$BATCH_SIZE" --hidden-dim "$HIDDEN_DIM" \
    --out "$out" \
    "${resume[@]}"
}

train_mappo_plus() {
  local seed="$1"
  local out="$RUN_ROOT/isaac_dual_mappo_plus_hard_seed${seed}"
  if [[ -f "$out/final.pt" ]]; then
    echo "[train] skip mappo_plus seed=$seed (final.pt exists)"
    return 0
  fi
  require_file "$DATA_ROOT/dagger_hard_seed${seed}/best_imitation.pt"
  local resume=()
  if [[ -f "$out/latest.pt" ]]; then
    resume=(--checkpoint "$out/latest.pt")
    echo "[train] resume mappo_plus seed=$seed from latest.pt"
  fi
  run_isaac_python python train_isaac_pursuit.py mappo --variant mappo_plus \
    --env-config "$HARD_CFG" \
    --actor-init "$DATA_ROOT/dagger_hard_seed${seed}/best_imitation.pt" \
    --demos "$DATA_ROOT/dagger_hard_seed${seed}/expert_episodes.pt" \
    --aux-demos "$DATA_ROOT/dagger_hard_seed${seed}/correction_episodes.pt" \
    --episodes "$EPISODES" --seed "$seed" \
    --budget-checkpoints "$BUDGET_CHECKPOINTS" \
    --batch-size "$BATCH_SIZE" --hidden-dim "$HIDDEN_DIM" \
    --out "$out" \
    "${resume[@]}"
}

train_matd3() {
  local seed="$1"
  local prior="$DATA_ROOT/hard_mpc_prior.pt"
  require_file "$DATA_ROOT/dagger_hard_seed${seed}/best_imitation.pt"
  require_file "$prior"
  local out="$RUN_ROOT/isaac_dual_A6_hard_seed${seed}"
  local resume=()
  if [[ -f "$out/final.pt" ]]; then
    echo "[train] skip A6 seed=${seed} (final.pt exists)"
    return 0
  fi
  if [[ -f "$out/latest.pt" ]]; then
    resume=(--checkpoint "$out/latest.pt")
    echo "[train] resume A6 seed=$seed from latest.pt"
  fi
  run_isaac_python python train_isaac_pursuit.py matd3 --variant A6 \
    --env-config "$HARD_CFG" \
    --actor-init "$DATA_ROOT/dagger_hard_seed${seed}/best_imitation.pt" \
    --prior "$prior" \
    --episodes "$EPISODES" --seed "$seed" \
    --budget-checkpoints "$BUDGET_CHECKPOINTS" \
    --batch-size "$BATCH_SIZE" --hidden-dim "$HIDDEN_DIM" \
    --out "$out" \
    "${resume[@]}"
}

phase_train_compare() {
  for seed in "${SEEDS[@]}"; do
    require_file "$DATA_ROOT/dagger_hard_seed${seed}/best_imitation.pt"
  done
  for seed in "${SEEDS[@]}"; do
    train_mappo_dagger "$seed"
    train_mappo_plus "$seed"
    train_matd3 "$seed"
  done
}

phase_train_ablation_key() {
  echo "[skip] train-ablation-key removed; kept variants are A6, mappo_plus, mappo_dagger"
}

phase_train_reward_ablation() {
  echo "[skip] train-reward-ablation removed with the unused reward-ablation configs"
}

eval_learned() {
  local method="$1"
  local tag="$2"
  local seed="$3"
  local ckpt="$RUN_ROOT/isaac_dual_${tag}_seed${seed}/final.pt"
  local out="$RUN_ROOT/isaac_dual_eval_${tag}_seed${seed}"
  require_file "$ckpt"
  if [[ -f "$out/summary.json" && "${EVAL_FORCE:-0}" != "1" ]]; then
    # Allow resume when summary exists but some requested levels are missing.
    local missing=0
    local level
    for level in $EVAL_LEVELS; do
      if [[ ! -f "$out/${level}.json" ]]; then
        missing=1
        break
      fi
    done
    if [[ "$missing" -eq 0 ]]; then
      echo "[eval] skip ${tag} seed=${seed} (summary.json complete for EVAL_LEVELS)"
      return 0
    fi
  fi
  # shellcheck disable=SC2086
  run_isaac_python python evaluate_isaac_pursuit.py --method "$method" \
    --checkpoint "$ckpt" \
    --levels $EVAL_LEVELS \
    --seed-start "$EVAL_SEED_START" --seed-count "$EVAL_SEED_COUNT" \
    --skip-existing-levels \
    --out "$out"
}

phase_eval_compare() {
  for seed in "${SEEDS[@]}"; do
    eval_learned mappo mappo_dagger_hard "$seed"
    eval_learned mappo mappo_plus_hard "$seed"
    eval_learned matd3 A6_hard "$seed"
  done
  # One Kit startup for all rule baselines.
  # shellcheck disable=SC2086
  run_isaac_python python evaluate_isaac_baselines.py \
    --methods apf frpn mpc \
    --levels $EVAL_LEVELS \
    --seed-start "$EVAL_SEED_START" --seed-count "$EVAL_SEED_COUNT" \
    --skip-existing-levels \
    --out-root "$RUN_ROOT"
}

phase_eval_ablation_key() {
  echo "[skip] eval-ablation-key removed; kept variants are A6, mappo_plus, mappo_dagger"
}

phase_eval_budgets() {
  for seed in "${SEEDS[@]}"; do
    run_isaac_python python evaluate_isaac_budget_curve.py --method mappo \
      --run-dir "$RUN_ROOT/isaac_dual_mappo_dagger_hard_seed${seed}" \
      --budgets "$BUDGET_CHECKPOINTS" \
      --seed-start "$EVAL_SEED_START" --seed-count "$BUDGET_EVAL_COUNT" \
      --out "$RUN_ROOT/isaac_dual_budget_mappo_dagger_seed${seed}"
    run_isaac_python python evaluate_isaac_budget_curve.py --method matd3 \
      --run-dir "$RUN_ROOT/isaac_dual_A6_hard_seed${seed}" \
      --budgets "$BUDGET_CHECKPOINTS" \
      --seed-start "$EVAL_SEED_START" --seed-count "$BUDGET_EVAL_COUNT" \
      --out "$RUN_ROOT/isaac_dual_budget_A6_seed${seed}"
  done
}

phase_extras() {
  # Information / capture / role diagnostics for paper figures (baselines, no weak-comm).
  mkdir -p "$RUN_ROOT/isaac_dual_extras"
  for method in mpc apf frpn; do
    run_isaac_python python evaluate_isaac_profile.py --method "$method" \
      --env-config "$MEDIUM_CFG" --tag "lidar_${method}" \
      --seed-start "$EVAL_SEED_START" --seed-count "$EXTRAS_SEED_COUNT" \
      --out "$RUN_ROOT/isaac_dual_extras/lidar_${method}"
  done
  run_isaac_python python evaluate_isaac_profile.py --method mpc \
    --env-config "$MEDIUM_CFG" --tag "dual_capture_mpc_roles" \
    --seed-start "$EVAL_SEED_START" --seed-count "$EXTRAS_SEED_COUNT" \
    --out "$RUN_ROOT/isaac_dual_extras/dual_capture_mpc_roles"
}

cmd="${1:-}"
case "$cmd" in
  env) print_env ;;
  data) activate_env; phase_data ;;
  dagger) activate_env; phase_dagger ;;
  train-compare) activate_env; phase_train_compare ;;
  train-ablation-key) activate_env; phase_train_ablation_key ;;
  train-reward-ablation) activate_env; phase_train_reward_ablation ;;
  eval-compare) activate_env; phase_eval_compare ;;
  eval-ablation-key) activate_env; phase_eval_ablation_key ;;
  eval-budgets) activate_env; phase_eval_budgets ;;
  extras) activate_env; phase_extras ;;
  all)
    activate_env
    phase_data
    phase_dagger
    phase_train_compare
    phase_train_ablation_key
    phase_train_reward_ablation
    phase_eval_compare
    phase_eval_budgets
    phase_eval_ablation_key
    phase_extras
    ;;
  *)
    cat <<EOF
Usage: bash $0 {env|data|dagger|train-compare|train-ablation-key|train-reward-ablation|eval-compare|eval-budgets|eval-ablation-key|extras|all}

Sample-efficiency protocol (default):
  EPISODES=$EPISODES
  BUDGET_CHECKPOINTS=$BUDGET_CHECKPOINTS
  SEEDS=${SEEDS[*]}

Notes:
  - ONE Isaac process per GPU.
  - A4c = clean visibility ablation (shared Medium prior); prefer over historical A4.
  - extras = oracle vs LiDAR, single vs dual capture (MPC), role counts, freshness off.
  - Weak communication is intentionally excluded.
EOF
    exit 1
    ;;
esac
