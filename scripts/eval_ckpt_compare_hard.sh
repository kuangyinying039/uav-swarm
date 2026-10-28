#!/usr/bin/env bash
# Hard-only held-out compare for physx_closed checkpoints (does not touch formal dirs).
# Usage: bash scripts/eval_ckpt_compare_hard.sh
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
PROJECT_RUNS="${UAV_PROJECT_RUNS:-/srv/data/projects/kyy/runs}"
TAG=physx_closed
TRAIN_ROOT="${PROJECT_RUNS}/isaac_dual_${TAG}/train"
OUT_ROOT="${TRAIN_ROOT}/isaac_dual_eval_ckpt_compare_hard"
LOG_DIR="${PROJECT_RUNS}/isaac_dual_${TAG}/logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"
LOG="${LOG_DIR}/eval_ckpt_compare_hard.log"

NUM_ENVS="${NUM_ENVS:-10}"
if [[ -f "${PROJECT_RUNS}/isaac_dual_${TAG}/num_envs.override" ]]; then
  NUM_ENVS="$(tr -d '[:space:]' < "${PROJECT_RUNS}/isaac_dual_${TAG}/num_envs.override")"
fi
SEED_START="${SEED_START:-7000000}"
SEED_COUNT="${SEED_COUNT:-100}"

source /opt/lab/apps/miniforge3-26.7.2/etc/profile.d/conda.sh
conda activate "${HOME}/.conda/envs/omnidrones-isaac41-writable"
export ISAACSIM_PATH=/opt/lab/apps/isaac-sim-4.1.0
set +u
# shellcheck disable=SC1091
source "${ISAACSIM_PATH}/setup_conda_env.sh"
set +u
export PYTHONPATH="/srv/data/projects/kyy/omnidrones:${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${HOME}/.cache"
export OMNI_USER_CACHE="${HOME}/.cache/isaac-kit"
unset DISPLAY || true
export OMNI_KIT_ACCEPT_EULA=YES
cd "${ROOT}"

eval_one() {
  local method="$1" ckpt="$2" name="$3"
  shift 3
  local extra=("$@")
  local out="${OUT_ROOT}/${name}"
  mkdir -p "$out"
  echo "[ckpt-compare] $(date -Is) ${name} method=${method} ckpt=${ckpt}" | tee -a "$LOG"
  python -u "${ROOT}/evaluate_isaac_pursuit.py" \
    --method "$method" \
    --checkpoint "$ckpt" \
    --levels hard \
    --no-skip-existing-levels \
    --physx-closed-loop \
    --no-native-lidar \
    --num-envs "${NUM_ENVS}" \
    --seed-start "${SEED_START}" \
    --seed-count "${SEED_COUNT}" \
    --log-every 10 \
    "${extra[@]}" \
    --out "$out" \
    | tee -a "$LOG"
}

A6_DIR="${TRAIN_ROOT}/isaac_dual_A6_mpc_hard_physx_closed_seed11"
MAPPO_DIR="${TRAIN_ROOT}/isaac_dual_mappo_plus_mpc_hard_physx_closed_seed11"
DAGGER="${PROJECT_RUNS}/isaac_dual_${TAG}/data/dagger_mpc_hard_seed11/best_imitation.pt"

eval_one matd3 "${A6_DIR}/best_capture.pt" a6_mpc_best_capture
eval_one matd3 "${A6_DIR}/final.pt" a6_mpc_final
# DAgger freeze is a MAPPO actor (algorithm=mappo_pursuit_3d_graph_v5), not MATD3.
eval_one mappo "${DAGGER}" dagger_mpc_best_imitation
eval_one mappo "${MAPPO_DIR}/best_capture.pt" mappo_mpc_best_capture
eval_one mappo "${MAPPO_DIR}/final.pt" mappo_mpc_final

# Summarize hard.json capture rates
python3 - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path
root = Path("/srv/data/projects/kyy/runs/isaac_dual_physx_closed/train/isaac_dual_eval_ckpt_compare_hard")
rows = []
for p in sorted(root.glob("*/hard.json")):
    s = json.loads(p.read_text())["summary"]
    rows.append((p.parent.name, s["capture_rate"], s.get("mean_censored_steps"), s.get("mean_return"), s.get("mean_collisions")))
print("=== hard held-out capture compare ===")
for name, rate, steps, ret, coll in rows:
    print(f"{name:32s}  rate={rate:.3f}  steps={steps:.1f}  ret={ret:.1f}  coll={coll:.2f}")
PY

echo "[ckpt-compare] $(date -Is) done" | tee -a "$LOG"
