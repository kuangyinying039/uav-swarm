#!/usr/bin/env bash
# Pure-RL ladder eval for FRPN-teacher residual-trained ckpts (NO blend at eval).
# Prior frpn_teacher_enhance only scored blend α=0.35; this fills α=1 / pure policy.
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
PROJECT_RUNS="${UAV_PROJECT_RUNS:-/srv/data/projects/kyy/runs}"
TAG=physx_closed
RUN_ROOT="${PROJECT_RUNS}/isaac_dual_${TAG}"
TRAIN_ROOT="${RUN_ROOT}/train"
LOG_DIR="${RUN_ROOT}/logs"
OUT_ROOT="${TRAIN_ROOT}/isaac_dual_eval_frpn_pure_rl"
LOG="${LOG_DIR}/frpn_pure_rl_eval.nohup"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

NUM_ENVS="${NUM_ENVS:-10}"
SEED_START="${SEED_START:-7000000}"
SEED_COUNT="${SEED_COUNT:-200}"
LEVELS=(nominal medium hard severe extreme)

CKPT_A6="${TRAIN_ROOT}/isaac_dual_A6_frpn_hard_${TAG}_seed11_residual_frpn_blend_ep400/best_capture.pt"
CKPT_MAPPO="${TRAIN_ROOT}/isaac_dual_mappo_plus_frpn_hard_${TAG}_seed11_residual_frpn_blend_ep400/best_capture.pt"

{
  echo "[frpn-pure-rl] $(date -Is) start n=${SEED_COUNT} levels=${LEVELS[*]} num_envs=${NUM_ENVS}"
  echo "[frpn-pure-rl] A6=${CKPT_A6}"
  echo "[frpn-pure-rl] MAPPO=${CKPT_MAPPO}"
} | tee -a "$LOG"

[[ -f "$CKPT_A6" ]] || { echo "missing $CKPT_A6" >&2; exit 1; }
[[ -f "$CKPT_MAPPO" ]] || { echo "missing $CKPT_MAPPO" >&2; exit 1; }

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

eval_pure() {
  local method="$1" ckpt="$2" name="$3"
  local out="${OUT_ROOT}/${name}"
  mkdir -p "$out"
  echo "[frpn-pure-rl] $(date -Is) START ${name} (pure RL, no residual)" | tee -a "$LOG"
  python -u "${ROOT}/evaluate_isaac_pursuit.py" \
    --method "$method" \
    --checkpoint "$ckpt" \
    --levels "${LEVELS[@]}" \
    --skip-existing-levels \
    --physx-closed-loop \
    --no-native-lidar \
    --num-envs "${NUM_ENVS}" \
    --seed-start "${SEED_START}" \
    --seed-count "${SEED_COUNT}" \
    --log-every 10 \
    --out "$out" \
    2>&1 | tee -a "$LOG"
  echo "[frpn-pure-rl] $(date -Is) DONE ${name}" | tee -a "$LOG"
}

eval_pure matd3 "$CKPT_A6" pure_rl_a6_frpn_teacher
eval_pure mappo "$CKPT_MAPPO" pure_rl_mappo_frpn_teacher

python3 - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path
root = Path("/srv/data/projects/kyy/runs/isaac_dual_physx_closed/train/isaac_dual_eval_frpn_pure_rl")
levels = ["nominal", "medium", "hard", "severe", "extreme"]
print("=== FRPN-teacher pure RL ladder @200 ===")
print(f"{'method':32s}", " ".join(f"{lv:>8s}" for lv in levels))
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    rates = []
    for lv in levels:
        p = d / f"{lv}.json"
        if not p.exists():
            rates.append(f"{'—':>8s}")
            continue
        s = json.loads(p.read_text())["summary"]
        rates.append(f"{s['capture_rate']:8.3f}")
    print(f"{d.name:32s}", " ".join(rates))
PY

echo "[frpn-pure-rl] $(date -Is) ALL COMPLETE" | tee -a "$LOG"
