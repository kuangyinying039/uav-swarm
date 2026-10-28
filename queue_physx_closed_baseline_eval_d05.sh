#!/usr/bin/env bash
# Formal baseline ladder under current physx_closed configs (target_diameter=0.5).
# Classical: APF / FRPN / MPC; plus frozen DAgger π0 as imitation reference.
# Does NOT evaluate old d=0.75 RL checkpoints.
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
PROJECT_RUNS="${UAV_PROJECT_RUNS:-/srv/data/projects/kyy/runs}"
TAG=physx_closed
RUN_ROOT="${PROJECT_RUNS}/isaac_dual_${TAG}"
TRAIN_ROOT="${RUN_ROOT}/train"
LOG_DIR="${RUN_ROOT}/logs"
OUT_ROOT="${TRAIN_ROOT}/isaac_dual_eval_baselines_d05"
mkdir -p "$LOG_DIR" "$OUT_ROOT"
LOG="${LOG_DIR}/baseline_eval_d05.log"

NUM_ENVS="${NUM_ENVS:-10}"
if [[ -f "${RUN_ROOT}/num_envs.override" ]]; then
  NUM_ENVS="$(tr -d '[:space:]' < "${RUN_ROOT}/num_envs.override")"
fi
SEED_START="${SEED_START:-7000000}"
SEED_COUNT="${SEED_COUNT:-100}"
LEVELS=(nominal medium hard severe extreme)
PI0="${RUN_ROOT}/data/dagger_mpc_hard_seed11/pi0_frozen.pt"

{
  echo "[baseline-d05] $(date -Is) start NUM_ENVS=${NUM_ENVS} seeds=${SEED_START}+${SEED_COUNT}"
  python3 - <<'PY'
import json
from pathlib import Path
root = Path("/srv/data/projects/kyy/xieton/repro/configs/pursuit_v2")
for level in ("nominal", "medium", "hard", "severe", "extreme"):
    p = root / f"radar5_dual_{level}_physx_closed.json"
    d = json.loads(p.read_text())
    assert float(d["target_diameter"]) == 0.5, (level, d["target_diameter"])
    print(f"[baseline-d05] {level}: diameter=0.5 target_speed={d.get('target_speed')} vmax={d.get('max_horizontal_velocity')}")
PY
} | tee -a "$LOG"

while pgrep -f 'train_isaac_pursuit.py|collect_isaac_pursuit_data.py|pretrain_isaac_pursuit_actor.py|evaluate_isaac_pursuit.py' >/dev/null 2>&1; do
  echo "[baseline-d05] $(date -Is) waiting for other Isaac jobs..." | tee -a "$LOG"
  sleep 30
done

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
  echo "[baseline-d05] $(date -Is) eval ${name} method=${method}" | tee -a "$LOG"
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

eval_one apf "" baseline_apf
eval_one frpn "" baseline_frpn
eval_one mpc "" baseline_mpc

if [[ -f "$PI0" ]]; then
  eval_one mappo "$PI0" pi0_dagger_frozen
else
  echo "[baseline-d05] WARN missing π0 at ${PI0}, skip imitation baseline" | tee -a "$LOG"
fi

python3 - <<PY | tee -a "$LOG"
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
print("=== baseline d05 snapshot (target_diameter=0.5) ===")
rows = []
for p in sorted(root.glob("*/*.json")):
    if p.name == "summary.json":
        continue
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    s = d.get("summary") or {}
    if "capture_rate" not in s:
        continue
    rows.append((p.parent.name, p.stem, float(s["capture_rate"]), float(s.get("mean_censored_steps") or 0),
                 s.get("target_diameter"), s.get("capture_radius")))
for name, level, rate, steps, diam, cap_r in rows:
    print(f"{name:28s} {level:8s} rate={rate:.3f} steps={steps:.1f} diam={diam} cap_r={cap_r}")
snap = {"protocol": {"target_diameter": 0.5, "physx_closed": True, "seed_start": ${SEED_START}, "seed_count": ${SEED_COUNT}},
        "rows": [{"method": n, "level": lv, "capture_rate": r, "mean_censored_steps": st, "target_diameter": d, "capture_radius": c}
                 for n, lv, r, st, d, c in rows]}
(root / "snapshot.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")
print(f"wrote {root / 'snapshot.json'}")
PY

echo "[baseline-d05] $(date -Is) complete" | tee -a "$LOG"
