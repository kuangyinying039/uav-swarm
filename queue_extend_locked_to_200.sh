#!/usr/bin/env bash
# Extend existing locked evals from 100 → 200 episodes (seeds 7000000..7000199).
# Reuses the first 100 rows via *.partial.jsonl resume; only runs the next 100.
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
RUN=/srv/data/projects/kyy/runs/isaac_dual_physx_closed
JOBS_JSON="${RUN}/logs/extend200_jobs.json"
LOG="${RUN}/logs/extend_locked_to_200.nohup"
NUM_ENVS="${NUM_ENVS:-10}"
SEED_START=7000000
SEED_COUNT=200

mkdir -p "${RUN}/logs"
exec > >(tee -a "$LOG") 2>&1
echo "[extend200] $(date -Is) start SEED_COUNT=${SEED_COUNT} NUM_ENVS=${NUM_ENVS}"

python3 - <<'PY'
"""Rebuild job list + seed ep100 json → partial.jsonl for resume."""
import json, shutil
from pathlib import Path

base = Path("/srv/data/projects/kyy/runs/isaac_dual_physx_closed/train")

def eval_method(raw: str) -> str:
    raw = str(raw)
    if raw.startswith("matd3"):
        return "matd3"
    if raw.startswith("mappo"):
        return "mappo"
    if raw in ("apf", "frpn", "mpc"):
        return raw
    return raw.split("+")[0]

def add_job(jobs, folder, name, levels, tag):
    hard = folder / name / "hard.json"
    d = json.loads(hard.read_text())
    s = d["summary"]
    jobs.append(
        {
            "tag": tag,
            "out": str(folder / name),
            "cli_method": eval_method(d.get("method") or s.get("method")),
            "checkpoint": d.get("checkpoint"),
            "residual_base": s.get("residual_base"),
            "residual_mode": s.get("residual_mode"),
            "residual_weight": s.get("residual_weight"),
            "levels": levels,
        }
    )
    out = folder / name
    for lv in levels:
        js = out / f"{lv}.json"
        if not js.exists():
            print(f"[prep] MISSING {out}/{lv}.json")
            continue
        payload = json.loads(js.read_text())
        seeds = list(map(int, payload.get("seeds") or []))
        if len(seeds) >= 200 and seeds == list(range(7000000, 7000200)):
            print(f"[prep] already 200: {tag}/{lv}")
            continue
        if len(payload.get("rows") or []) != 100:
            print(f"[prep] unexpected n={len(payload.get('rows') or [])} for {tag}/{lv}")
        bak = out / f"{lv}.ep100.json"
        if not bak.exists():
            shutil.copy2(js, bak)
        partial = out / f"{lv}.partial.jsonl"
        with partial.open("w", encoding="utf-8") as stream:
            for row in payload["rows"]:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[prep] {tag}/{lv}: wrote {len(payload['rows'])} partial rows → {partial.name}")

jobs = []
lf = base / "isaac_dual_eval_locked_full"
for name in sorted(p.name for p in lf.iterdir() if p.is_dir()):
    add_job(jobs, lf, name, ["nominal", "medium", "hard", "severe", "extreme"], f"locked_full/{name}")

fe = base / "isaac_dual_eval_frpn_teacher_enhance"
for name in ["pi0_dagger_frozen", "residual_a6_frpn_blend", "residual_mappo_frpn_blend"]:
    add_job(jobs, fe, name, ["nominal", "medium", "hard", "severe", "extreme"], f"frpn_eval/{name}")

sw = base / "isaac_dual_eval_alpha_sweep_hard"
for name in sorted(p.name for p in sw.iterdir() if p.is_dir()):
    add_job(jobs, sw, name, ["hard"], f"alpha/{name}")

Path("/srv/data/projects/kyy/runs/isaac_dual_physx_closed/logs/extend200_jobs.json").write_text(
    json.dumps(jobs, indent=2)
)
print(f"[prep] jobs={len(jobs)}")
PY

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
cd "$ROOT"

python3 - <<'PY'
import json, subprocess, sys
from pathlib import Path

jobs = json.loads(Path("/srv/data/projects/kyy/runs/isaac_dual_physx_closed/logs/extend200_jobs.json").read_text())
root = Path("/srv/data/projects/kyy/xieton/repro")
num_envs = "10"
seed_start = "7000000"
seed_count = "200"

for i, job in enumerate(jobs, 1):
    out = Path(job["out"])
    # Skip if all levels already at 200
    levels = job["levels"]
    done = True
    for lv in levels:
        p = out / f"{lv}.json"
        if not p.exists():
            done = False
            break
        payload = json.loads(p.read_text())
        if list(map(int, payload.get("seeds") or [])) != list(range(7000000, 7000200)):
            done = False
            break
    if done:
        print(f"[extend200] ({i}/{len(jobs)}) SKIP already-200 {job['tag']}", flush=True)
        continue

    args = [
        "python", "-u", str(root / "evaluate_isaac_pursuit.py"),
        "--method", job["cli_method"],
        "--levels", *levels,
        "--skip-existing-levels",
        "--physx-closed-loop",
        "--no-native-lidar",
        "--num-envs", num_envs,
        "--seed-start", seed_start,
        "--seed-count", seed_count,
        "--log-every", "10",
        "--out", str(out),
    ]
    if job.get("checkpoint"):
        args += ["--checkpoint", job["checkpoint"]]
    if job.get("residual_base"):
        args += [
            "--residual-base", str(job["residual_base"]),
            "--residual-mode", str(job["residual_mode"] or "blend"),
            "--residual-weight", str(job["residual_weight"]),
        ]
    print(f"[extend200] ({i}/{len(jobs)}) START {job['tag']}", flush=True)
    print("[extend200] cmd:", " ".join(args), flush=True)
    rc = subprocess.call(args)
    if rc != 0:
        print(f"[extend200] FAILED {job['tag']} rc={rc}", flush=True)
        sys.exit(rc)
    # quick hard rate
    hard = json.loads((out / "hard.json").read_text())["summary"]
    print(
        f"[extend200] DONE {job['tag']} hard={hard['capture_rate']:.3f} "
        f"episodes={hard['episodes']}",
        flush=True,
    )

print("[extend200] ALL JOBS COMPLETE", flush=True)

# Snapshot tables
base = Path("/srv/data/projects/kyy/runs/isaac_dual_physx_closed/train")
levels = ["nominal", "medium", "hard", "severe", "extreme"]

def rate(folder, name, lv="hard"):
    p = folder / name / f"{lv}.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())["summary"]
    return s["capture_rate"], s["episodes"]

print("=== locked_full @200 ===")
lf = base / "isaac_dual_eval_locked_full"
print(f"{'method':34s}", " ".join(f"{lv:>8s}" for lv in levels), "  n")
for name in sorted(p.name for p in lf.iterdir() if p.is_dir()):
    cells = []
    n = None
    for lv in levels:
        r = rate(lf, name, lv)
        if r is None:
            cells.append(f"{'—':>8s}")
        else:
            cells.append(f"{r[0]:8.3f}")
            n = r[1]
    print(f"{name:34s}", " ".join(cells), f" n={n}")

print("=== frpn unique @200 ===")
fe = base / "isaac_dual_eval_frpn_teacher_enhance"
for name in ["pi0_dagger_frozen", "residual_a6_frpn_blend", "residual_mappo_frpn_blend"]:
    cells = []
    n = None
    for lv in levels:
        r = rate(fe, name, lv)
        if r is None:
            cells.append(f"{'—':>8s}")
        else:
            cells.append(f"{r[0]:8.3f}")
            n = r[1]
    print(f"{name:34s}", " ".join(cells), f" n={n}")

print("=== alpha sweep hard @200 ===")
sw = base / "isaac_dual_eval_alpha_sweep_hard"
for name in sorted(p.name for p in sw.iterdir() if p.is_dir()):
    r = rate(sw, name, "hard")
    print(f"{name:34s} hard={r[0]:.3f} n={r[1]}" if r else name)
PY

echo "[extend200] $(date -Is) complete"
