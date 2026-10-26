#!/usr/bin/env bash
# Improved RL recipes to beat classical Dual-Medium teachers (MPC≈0.89, FRPN≈0.85).
#
# MATD3 A6: O2O stable (TD3+BC + actor anchor) + critic pretrain + UTD=4 + MPC blend
# MAPPO mappo_plus: stronger demo retention + return norm + held-out best_capture
#
# Usage:
#   bash run_isaac_rl_improved.sh matd3-a6 [seed=11] [num_envs=16] [episodes=1000]
#   bash run_isaac_rl_improved.sh mappo-plus [seed=11] [num_envs=16] [episodes=1000]
#   bash run_isaac_rl_improved.sh eval-matd3 RUN_DIR
#   bash run_isaac_rl_improved.sh eval-mappo RUN_DIR
# Override default parallel width with ISAAC_NUM_ENVS.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="${RUN_ROOT:-$HOME/runs}"
DATA="${DATA_ROOT:-$ROOT/isaac_dual_data}"
MEDIUM="configs/pursuit_v2/radar5_dual_medium.json"
CMD="${1:-}"
SEED="${2:-11}"
NUM_ENVS="${3:-${ISAAC_NUM_ENVS:-16}}"
EPISODES="${4:-1000}"

ACTOR="${DATA}/dagger_seed${SEED}/best_imitation.pt"
DEMOS="${DATA}/dagger_seed${SEED}/expert_episodes.pt"
PRIOR="${DATA}/medium_mpc_prior.pt"
AUX="${DATA}/dagger_seed${SEED}/correction_episodes.pt"

case "$CMD" in
  matd3-a6)
    OUT="$ROOT/isaac_dual_A6_seed${SEED}_n${NUM_ENVS}"
    mkdir -p "$OUT"
    python train_isaac_pursuit.py matd3 \
      --variant A6 \
      --num-envs "$NUM_ENVS" \
      --env-config "$MEDIUM" \
      --actor-init "$ACTOR" \
      --prior "$PRIOR" \
      --episodes "$EPISODES" \
      --seed "$SEED" \
      --validation-episodes 20 \
      --out "$OUT" \
      2>&1 | tee "$OUT/train.log"
    ;;
  mappo-plus)
    OUT="$ROOT/isaac_dual_mappo_plus_seed${SEED}_n${NUM_ENVS}"
    mkdir -p "$OUT"
    python train_isaac_pursuit.py mappo \
      --variant mappo_plus \
      --num-envs "$NUM_ENVS" \
      --env-config "$MEDIUM" \
      --actor-init "$ACTOR" \
      --demos "$DEMOS" \
      --aux-demos "$AUX" \
      --episodes "$EPISODES" \
      --seed "$SEED" \
      --validation-episodes 20 \
      --out "$OUT" \
      2>&1 | tee "$OUT/train.log"
    ;;
  eval-matd3|eval-mappo)
    RUN_DIR="${2:?run directory}"
    METHOD="matd3"
    [[ "$CMD" == eval-mappo ]] && METHOD="mappo"
    CKPT="$RUN_DIR/best_capture.pt"
    [[ -f "$CKPT" ]] || CKPT="$RUN_DIR/final.pt"
    OUT="$ROOT/isaac_dual_eval_$(basename "$RUN_DIR")"
    EXTRA=()
    if [[ "$METHOD" == matd3 ]]; then
      # A6 anneals to pure RL; evaluate the learned policy without classical mix
      # unless RESIDUAL_EVAL_WEIGHT is set (e.g. 0.4 for mid-training hybrids).
      if [[ -n "${RESIDUAL_EVAL_WEIGHT:-}" ]]; then
        EXTRA=(--residual-base mpc --residual-mode blend --residual-weight "$RESIDUAL_EVAL_WEIGHT")
      fi
    fi
    python evaluate_isaac_profile.py \
      --method "$METHOD" \
      --checkpoint "$CKPT" \
      --levels medium \
      --seed-count 200 \
      --out "$OUT" \
      "${EXTRA[@]}" \
      2>&1 | tee "$OUT/eval.log"
    ;;
  *)
    echo "Unknown command: $CMD" >&2
    echo "Use: matd3-a6 | mappo-plus | eval-matd3 | eval-mappo" >&2
    exit 2
    ;;
esac
