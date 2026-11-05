#!/usr/bin/env bash
# Full locked ladder eval (nominal→extreme) after residual parallel-env fix.
# Re-scores baselines + π0 + residual (blend) + pure-RL archive under one protocol.
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
PROJECT_RUNS="${UAV_PROJECT_RUNS:-/srv/data/projects/kyy/runs}"
TAG=physx_closed
RUN_ROOT="${PROJECT_RUNS}/isaac_dual_${TAG}"
TRAIN_ROOT="${RUN_ROOT}/train"
DATA_ROOT="${RUN_ROOT}/data"
LOG_DIR="${RUN_ROOT}/logs"
OUT_ROOT="${TRAIN_ROOT}/isaac_dual_eval_locked_full"
LOG="${LOG_DIR}/locked_full_eval.log"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

NUM_ENVS="${NUM_ENVS:-10}"
SEED_START="${SEED_START:-7000000}"
SEED_COUNT="${SEED_COUNT:-100}"
LEVELS=(nominal medium hard severe extreme)
RESIDUAL_WEIGHT="${RESIDUAL_WEIGHT:-0.25}"
# Prefer blend for MAPPO full-action ckpts (see train --residual-mode).
MAPPO_RESIDUAL_MODE="${MAPPO_RESIDUAL_MODE:-blend}"

source /opt/lab/apps/miniforge3-26.7.2/etc/profile.d/conda.sh
conda activate "${HOME}/.conda/envs/omnidrones-isaac41-writable"
export ISAACSIM_PATH=/opt/lab/apps/isaac-sim-4.1.0
set +u
# shellcheck disable=SC1091
source "${ISAACSIM_PATH}/setup_conda_env.sh"
set -euo pipefail
export PYTHONPATH="/srv/data/projects/kyy/omnidrones:${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${HOME}/.cache"
export OMNI_USER_CACHE="${HOME}/.cache/isaac-kit"
unset DISPLAY || true
export OMNI_KIT_ACCEPT_EULA=YES
cd "${ROOT}"

{
  echo "[locked-full] $(date -Is) start NUM_ENVS=${NUM_ENVS} seeds=${SEED_START}+${SEED_COUNT}"
  echo "[locked-full] levels=${LEVELS[*]} mappo_mode=${MAPPO_RESIDUAL_MODE} w=${RESIDUAL_WEIGHT}"
} | tee -a "$LOG"

eval_one() {
  local method="$1" ckpt="$2" name="$3"
  shift 3
  local extra=("$@")
  local out="${OUT_ROOT}/${name}"
  mkdir -p "$out"
  echo "[locked-full] $(date -Is) eval ${name}" | tee -a "$LOG"
  local args=(
    -u "${ROOT}/evaluate_isaac_pursuit.py"
    --method "$method"
    --levels "${LEVELS[@]}"
    --no-skip-existing-levels
    --physx-closed-loop
    --no-native-lidar
    --num-envs "${NUM_ENVS}"
    --seed-start "${SEED_START}"
    --seed-count "${SEED_COUNT}"
    --log-every 10
    --out "$out"
  )
  if [[ -n "$ckpt" ]]; then
    args+=(--checkpoint "$ckpt")
  fi
  args+=("${extra[@]}")
  python "${args[@]}" 2>&1 | tee -a "$LOG"
}

PI0="${DATA_ROOT}/dagger_mpc_hard_seed11/pi0_frozen.pt"
RES_A6="${TRAIN_ROOT}/isaac_dual_A6_mpc_hard_${TAG}_seed11_residual_mpc_zeroinit_ep200/best_capture.pt"
RES_MAPPO="${TRAIN_ROOT}/isaac_dual_mappo_plus_mpc_hard_${TAG}_seed11_residual_mpc_zeroinit_ep200/best_capture.pt"
ARCH_A6="${TRAIN_ROOT}/isaac_dual_A6_mpc_hard_${TAG}_seed11/best_capture.pt"
ARCH_MAPPO="${TRAIN_ROOT}/isaac_dual_mappo_plus_mpc_hard_${TAG}_seed11/best_capture.pt"

eval_one apf "" baseline_apf
eval_one frpn "" baseline_frpn
eval_one mpc "" baseline_mpc

if [[ -f "$PI0" ]]; then
  eval_one mappo "$PI0" pi0_dagger_frozen
fi

if [[ -f "$RES_A6" ]]; then
  eval_one matd3 "$RES_A6" residual_a6_mpc_blend \
    --residual-base mpc --residual-mode blend --residual-weight "${RESIDUAL_WEIGHT}"
  # Teacher floor check: w=0 must track MPC
  eval_one matd3 "$RES_A6" residual_a6_mpc_w0 \
    --residual-base mpc --residual-mode blend --residual-weight 0.0
fi

if [[ -f "$RES_MAPPO" ]]; then
  eval_one mappo "$RES_MAPPO" residual_mappo_mpc_blend \
    --residual-base mpc --residual-mode "${MAPPO_RESIDUAL_MODE}" --residual-weight "${RESIDUAL_WEIGHT}"
  eval_one mappo "$RES_MAPPO" residual_mappo_mpc_w0 \
    --residual-base mpc --residual-mode "${MAPPO_RESIDUAL_MODE}" --residual-weight 0.0
fi

if [[ -f "$ARCH_A6" ]]; then
  eval_one matd3 "$ARCH_A6" pure_rl_a6_best_archive
fi
if [[ -f "$ARCH_MAPPO" ]]; then
  eval_one mappo "$ARCH_MAPPO" pure_rl_mappo_best_archive
fi

python3 - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path
root = Path("/srv/data/projects/kyy/runs/isaac_dual_physx_closed/train/isaac_dual_eval_locked_full")
levels = ["nominal", "medium", "hard", "severe", "extreme"]
print("=== locked-full snapshot ===")
print(f"{'method':36s}", " ".join(f"{lv:>8s}" for lv in levels))
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    rates = []
    for lv in levels:
        p = d / f"{lv}.json"
        if not p.exists():
            rates.append(f"{'—':>8s}")
            continue
        s = json.loads(p.read_text())["summary"]
        rates.append(f"{s['capture_rate']:8.3f}")
    print(f"{d.name:36s}", " ".join(rates))
print("wrote under", root)
PY

echo "[locked-full] $(date -Is) complete" | tee -a "$LOG"
