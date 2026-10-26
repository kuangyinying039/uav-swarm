#!/usr/bin/env bash
# Paper mainline single-seed loop:
#   wait A6 MATD3 → Isaac Medium eval → MAPPO-plus train → Isaac Medium eval → summary
#
# Usage:
#   bash run_isaac_mainline_pipeline.sh          # full chain (assumes A6 already running/done)
#   bash run_isaac_mainline_pipeline.sh wait-a6  # only wait
#   bash run_isaac_mainline_pipeline.sh eval-a6
#   bash run_isaac_mainline_pipeline.sh train-mappo
#   bash run_isaac_mainline_pipeline.sh eval-mappo
#   bash run_isaac_mainline_pipeline.sh summarize
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
cd "$ROOT"

SEED="${SEED:-11}"
EPISODES="${EPISODES:-1000}"
EVAL_SEEDS="${EVAL_SEEDS:-200}"
EVAL_SEED_START="${EVAL_SEED_START:-7000000}"
# Full paper protocol: all dual difficulty tiers (not medium-only).
EVAL_LEVELS="${EVAL_LEVELS:-nominal medium hard extreme}"
A6_OUT="${A6_OUT:-$HOME/runs/isaac_dual_A6_seed11_n4}"
MAPPO_OUT="${MAPPO_OUT:-$HOME/runs/isaac_dual_mappo_plus_seed${SEED}}"
COMPARE_OUT="${COMPARE_OUT:-$HOME/runs/isaac_dual_mainline_compare_seed${SEED}}"
DATA="${DATA_ROOT:-$HOME/runs/isaac_dual_data}"
MEDIUM="configs/pursuit_v2/radar5_dual_medium.json"
LOGDIR="${LOGDIR:-$HOME/runs/isaac_dual_logs}"
mkdir -p "$LOGDIR" "$COMPARE_OUT"

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
  export PYTHONPATH="/srv/data/projects/kyy/omnidrones:$ROOT:${PYTHONPATH:-}"
  export PYTHONUNBUFFERED=1
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
  mkdir -p "$HOME/.cache/isaac-kit" "$HOME/.cache/ov"
  export OMNI_USER_CACHE="${OMNI_USER_CACHE:-$HOME/.cache/isaac-kit}"
  unset DISPLAY || true
  export OMNI_KIT_ACCEPT_EULA=YES
}

run_isaac_python() {
  local attempt=1 max_attempts=3 rc=0
  while (( attempt <= max_attempts )); do
    echo "[isaac-run] attempt=${attempt}/${max_attempts}: $*"
    set +e
    "$@"
    rc=$?
    set -e
    (( rc == 0 )) && return 0
    if (( rc == 139 || rc == 134 || rc == 132 )); then
      echo "[isaac-run] Kit crash rc=$rc; retry" >&2
      rm -f "${XDG_CACHE_HOME}/ov/_cache.lock" "${OMNI_USER_CACHE}/_cache.lock" 2>/dev/null || true
      sleep $(( attempt * 20 ))
      attempt=$((attempt + 1))
      continue
    fi
    return "$rc"
  done
  return "$rc"
}

pick_ckpt() {
  local run_dir="$1"
  if [[ -f "$run_dir/best_capture.pt" ]]; then
    echo "$run_dir/best_capture.pt"
  elif [[ -f "$run_dir/final.pt" ]]; then
    echo "$run_dir/final.pt"
  else
    echo ""
  fi
}

wait_a6() {
  echo "[pipeline] waiting for A6 final.pt under $A6_OUT ..."
  local log
  log="$(ls -t "$LOGDIR"/train_A6_seed11_n4_*.log 2>/dev/null | head -1 || true)"
  while true; do
    if [[ -f "$A6_OUT/final.pt" ]]; then
      echo "[pipeline] A6 finished: $A6_OUT/final.pt"
      return 0
    fi
    if ! pgrep -f 'train_isaac_pursuit.py matd3 --variant A6' >/dev/null 2>&1; then
      # brief grace for final.pt flush
      sleep 15
      if [[ -f "$A6_OUT/final.pt" ]]; then
        return 0
      fi
      echo "[pipeline] A6 process gone but no final.pt — check logs" >&2
      ls -la "$A6_OUT" || true
      return 1
    fi
    if [[ -n "$log" && -f "$log" ]]; then
      rg -n 'matd3-vec\] episode=' "$log" | tail -1 || true
    fi
    sleep 120
  done
}

eval_a6() {
  activate_env
  local ckpt
  ckpt="$(pick_ckpt "$A6_OUT")"
  [[ -n "$ckpt" ]] || { echo "no A6 checkpoint"; return 1; }
  # shellcheck disable=SC2206
  local levels=( $EVAL_LEVELS )
  local out="$HOME/runs/isaac_dual_eval_A6_seed${SEED}"
  mkdir -p "$out"
  echo "[pipeline] eval A6 pure RL ckpt=$ckpt levels=${levels[*]}"
  run_isaac_python python evaluate_isaac_pursuit.py \
    --method matd3 \
    --checkpoint "$ckpt" \
    --levels "${levels[@]}" \
    --seed-start "$EVAL_SEED_START" \
    --seed-count "$EVAL_SEEDS" \
    --skip-existing-levels \
    --out "$out" \
    2>&1 | tee -a "$out/eval_pure.log"
  local out_h="$HOME/runs/isaac_dual_eval_A6_seed${SEED}_mpcblend"
  mkdir -p "$out_h"
  echo "[pipeline] eval A6 + MPC blend w=0.35 levels=${levels[*]}"
  run_isaac_python python evaluate_isaac_pursuit.py \
    --method matd3 \
    --checkpoint "$ckpt" \
    --levels "${levels[@]}" \
    --seed-start "$EVAL_SEED_START" \
    --seed-count "$EVAL_SEEDS" \
    --residual-base mpc --residual-mode blend --residual-weight 0.35 \
    --skip-existing-levels \
    --out "$out_h" \
    2>&1 | tee -a "$out_h/eval_blend.log"
}

train_mappo() {
  activate_env
  mkdir -p "$MAPPO_OUT"
  echo "[pipeline] train mappo_plus seed=$SEED episodes=$EPISODES"
  run_isaac_python python train_isaac_pursuit.py mappo \
    --variant mappo_plus \
    --env-config "$MEDIUM" \
    --actor-init "$DATA/dagger_seed${SEED}/best_imitation.pt" \
    --demos "$DATA/dagger_seed${SEED}/expert_episodes.pt" \
    --aux-demos "$DATA/dagger_seed${SEED}/correction_episodes.pt" \
    --episodes "$EPISODES" \
    --seed "$SEED" \
    --validation-episodes 20 \
    --budget-checkpoints 250,500,750,1000 \
    --out "$MAPPO_OUT" \
    2>&1 | tee "$MAPPO_OUT/train.log"
}

eval_mappo() {
  activate_env
  local ckpt
  ckpt="$(pick_ckpt "$MAPPO_OUT")"
  [[ -n "$ckpt" ]] || { echo "no MAPPO checkpoint"; return 1; }
  # shellcheck disable=SC2206
  local levels=( $EVAL_LEVELS )
  local out="$HOME/runs/isaac_dual_eval_mappo_plus_seed${SEED}"
  mkdir -p "$out"
  echo "[pipeline] eval mappo_plus pure ckpt=$ckpt levels=${levels[*]}"
  run_isaac_python python evaluate_isaac_pursuit.py \
    --method mappo \
    --checkpoint "$ckpt" \
    --levels "${levels[@]}" \
    --seed-start "$EVAL_SEED_START" \
    --seed-count "$EVAL_SEEDS" \
    --skip-existing-levels \
    --out "$out" \
    2>&1 | tee -a "$out/eval_pure.log"
  local out_h="$HOME/runs/isaac_dual_eval_mappo_plus_seed${SEED}_mpcblend"
  mkdir -p "$out_h"
  echo "[pipeline] eval mappo_plus + MPC blend w=0.35 levels=${levels[*]}"
  run_isaac_python python evaluate_isaac_pursuit.py \
    --method mappo \
    --checkpoint "$ckpt" \
    --levels "${levels[@]}" \
    --seed-start "$EVAL_SEED_START" \
    --seed-count "$EVAL_SEEDS" \
    --residual-base mpc --residual-mode blend --residual-weight 0.35 \
    --skip-existing-levels \
    --out "$out_h" \
    2>&1 | tee -a "$out_h/eval_blend.log"
}

summarize() {
  SEED="$SEED" EVAL_LEVELS="$EVAL_LEVELS" python - <<'PY'
import json
import os
from pathlib import Path

home = Path.home() / "runs"
seed = int(os.environ.get("SEED", "11"))
levels = os.environ.get("EVAL_LEVELS", "nominal medium hard extreme").split()
methods = [
    ("mpc", f"isaac_dual_eval_mpc"),
    ("frpn", f"isaac_dual_eval_frpn"),
    ("apf", f"isaac_dual_eval_apf"),
    ("A5_old", f"isaac_dual_eval_A5_seed{seed}"),
    ("mappo_dagger", f"isaac_dual_eval_mappo_dagger_seed{seed}"),
    ("A6_pure", f"isaac_dual_eval_A6_seed{seed}"),
    ("A6_mpcblend", f"isaac_dual_eval_A6_seed{seed}_mpcblend"),
    ("A6_rlblend065", f"isaac_dual_eval_A6_seed{seed}_rlblend065"),
    ("mappo_plus_pure", f"isaac_dual_eval_mappo_plus_seed{seed}"),
    ("mappo_plus_mpcblend", f"isaac_dual_eval_mappo_plus_seed{seed}_mpcblend"),
    ("mappo_plus_rlblend065", f"isaac_dual_eval_mappo_plus_seed{seed}_rlblend065"),
]
by_level = {}
for level in levels:
    rows = []
    for name, folder in methods:
        path = home / folder / f"{level}.json"
        if not path.is_file():
            rows.append({"method": name, "capture_rate": None, "missing": True})
            continue
        s = json.loads(path.read_text()).get("summary", {})
        rows.append({
            "method": name,
            "capture_rate": s.get("capture_rate"),
            "mean_return": s.get("mean_return"),
            "mean_censored_steps": s.get("mean_censored_steps"),
            "wilson_95": s.get("capture_rate_wilson_95"),
            "missing": False,
        })
    by_level[level] = rows

out = home / f"isaac_dual_mainline_compare_seed{seed}"
out.mkdir(parents=True, exist_ok=True)
payload = {"seed": seed, "levels": levels, "episodes_eval": 200, "by_level": by_level}
(out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = ["# Mainline compare (seed %d, 200 eps/level)" % seed, ""]
for level in levels:
    lines.append(f"## {level}")
    lines.append("| method | capture_rate | mean_return |")
    lines.append("|---|---:|---:|")
    for r in by_level[level]:
        cr = "—" if r["capture_rate"] is None else f"{r['capture_rate']:.3f}"
        mr = "—" if r.get("mean_return") is None else f"{r['mean_return']:.1f}"
        lines.append(f"| {r['method']} | {cr} | {mr} |")
    lines.append("")
text = "\n".join(lines)
(out / "summary.md").write_text(text + "\n", encoding="utf-8")
print(text)
print(f"\n[pipeline] wrote {out / 'summary.json'}")
PY
}

wait_mappo_and_gpu() {
  echo "[pipeline] waiting for MAPPO final.pt under $MAPPO_OUT ..."
  while [[ ! -f "$MAPPO_OUT/final.pt" ]]; do
    if ! pgrep -f 'train_isaac_pursuit.py mappo --variant mappo_plus' >/dev/null 2>&1; then
      sleep 20
      [[ -f "$MAPPO_OUT/final.pt" ]] && break
      echo "[pipeline] MAPPO process gone without final.pt" >&2
      ls -la "$MAPPO_OUT" || true
      return 1
    fi
    if [[ -f "$MAPPO_OUT/train.log" ]]; then
      rg -n '\[MAPPO\] episode=' "$MAPPO_OUT/train.log" | tail -1 || true
    fi
    sleep 120
  done
  echo "[pipeline] MAPPO finished; waiting for GPU free (no train/eval) ..."
  # Let the older medium-only mainline eval finish if still running.
  while pgrep -f 'train_isaac_pursuit.py|evaluate_isaac_pursuit.py' >/dev/null 2>&1; do
    pgrep -af 'train_isaac_pursuit.py|evaluate_isaac_pursuit.py' | head -5 || true
    sleep 60
  done
  echo "[pipeline] GPU free — starting full-level evals"
}

full_eval() {
  # Completes nominal/hard/extreme (and any missing medium) for A6 + mappo_plus.
  EVAL_LEVELS="${EVAL_LEVELS:-nominal medium hard extreme}"
  wait_mappo_and_gpu
  eval_a6
  eval_mappo
  SEED="$SEED" EVAL_LEVELS="$EVAL_LEVELS" summarize
  echo "[pipeline] FULL-LEVEL EVAL DONE $(date)"
}

cmd="${1:-all}"
case "$cmd" in
  wait-a6) wait_a6 ;;
  eval-a6) eval_a6 ;;
  train-mappo) train_mappo ;;
  eval-mappo) eval_mappo ;;
  summarize) SEED="$SEED" EVAL_LEVELS="${EVAL_LEVELS:-nominal medium hard extreme}" summarize ;;
  full-eval) full_eval ;;
  all)
    wait_a6
    eval_a6
    train_mappo
    eval_mappo
    SEED="$SEED" EVAL_LEVELS="${EVAL_LEVELS:-nominal medium hard extreme}" summarize
    echo "[pipeline] ALL DONE $(date)"
    ;;
  *)
    echo "usage: $0 {all|wait-a6|eval-a6|train-mappo|eval-mappo|full-eval|summarize}" >&2
    exit 2
    ;;
esac
