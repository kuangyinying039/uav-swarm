#!/usr/bin/env python3
"""Offline O2O collapse diagnosis: π0 / best / final from saved artifacts.

Produces:
  - validation capture curve vs episode
  - summary table: init (π0), best, final
  - optional action-shift / critic diagnostics from history if present

Does **not** claim RL gain by itself — compare π_best vs π0 on a locked test set
(see scripts/eval_ckpt_compare_hard.sh with method=mappo for DAgger π0).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _matd3_summary(train_dir: Path) -> dict:
    val = _load_json(train_dir / "validation.json") or []
    rows = [r for r in val if isinstance(r, dict)]
    pre = [r for r in rows if int(r.get("episode", -1)) == 0 or r.get("pre_rl")]
    considered = [r for r in rows if r.get("considered_for_best")]
    best = max(considered, key=lambda r: (r.get("capture_rate", -1), -r.get("mean_censored_steps", 1e9)), default=None)
    last = rows[-1] if rows else None
    return {
        "n_validations": len(rows),
        "pi0_capture": None if not pre else pre[0].get("capture_rate"),
        "pi0_episode": None if not pre else pre[0].get("episode"),
        "best_capture": None if best is None else best.get("capture_rate"),
        "best_episode": None if best is None else best.get("episode"),
        "final_capture": None if last is None else last.get("capture_rate"),
        "final_episode": None if last is None else last.get("episode"),
        "curve": [
            {
                "episode": r.get("episode"),
                "capture_rate": r.get("capture_rate"),
                "mean_return": r.get("mean_return"),
                "considered_for_best": r.get("considered_for_best"),
                "actor_rollback": r.get("actor_rollback"),
            }
            for r in rows
        ],
    }


def _mappo_summary(train_dir: Path) -> dict:
    val = _load_json(train_dir / "validation.json") or []
    rows = [r for r in val if isinstance(r, dict)]
    best = max(rows, key=lambda r: (r.get("capture_rate", -1), -r.get("mean_censored_steps", 1e9)), default=None)
    last = rows[-1] if rows else None
    first = rows[0] if rows else None
    hist = _load_json(train_dir / "history.json") or []
    # Proxy action-shift / health from pursuit history if logged.
    shifts = []
    for row in hist:
        if not isinstance(row, dict):
            continue
        if "exact_policy_kl" in row or "action_mean_saturation" in row:
            shifts.append(
                {
                    "episode": row.get("episode"),
                    "exact_policy_kl": row.get("exact_policy_kl"),
                    "action_mean_saturation": row.get("action_mean_saturation"),
                }
            )
    return {
        "n_validations": len(rows),
        "pi0_capture": None if first is None else first.get("capture_rate"),
        "pi0_episode": None if first is None else first.get("episode"),
        "best_capture": None if best is None else best.get("capture_rate"),
        "best_episode": None if best is None else best.get("episode"),
        "final_capture": None if last is None else last.get("capture_rate"),
        "final_episode": None if last is None else last.get("episode"),
        "curve": [
            {
                "episode": r.get("episode"),
                "capture_rate": r.get("capture_rate"),
                "mean_return": r.get("mean_return"),
            }
            for r in rows
        ],
        "policy_health": shifts[-50:],
    }


def _interpret(summary: dict) -> list[str]:
    notes = []
    pi0 = summary.get("pi0_capture")
    best = summary.get("best_capture")
    final = summary.get("final_capture")
    if best is None:
        notes.append("no validation best found")
        return notes
    if final is not None and best > final + 1e-6:
        notes.append("best >> final → non-monotonic O2O (collapse / forgetting risk)")
    if pi0 is not None and best <= pi0 + 1e-6:
        notes.append("best ≯ π0 → cannot claim online RL gain yet (need locked test vs π0)")
    if pi0 is not None and best > pi0 + 1e-6:
        notes.append("best > π0 on train-val → candidate RL gain; confirm on locked test + multi-seed")
    if pi0 is not None and final is not None and final < pi0:
        notes.append("final < π0 → late updates likely damaged the initialization")
    return notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--a6-dir",
        type=Path,
        default=Path(
            "/srv/data/projects/kyy/runs/isaac_dual_physx_closed/train/"
            "isaac_dual_A6_mpc_hard_physx_closed_seed11"
        ),
    )
    parser.add_argument(
        "--mappo-dir",
        type=Path,
        default=Path(
            "/srv/data/projects/kyy/runs/isaac_dual_physx_closed/train/"
            "isaac_dual_mappo_plus_mpc_hard_physx_closed_seed11"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/srv/data/projects/kyy/runs/isaac_dual_physx_closed/train/o2o_diagnosis.json"
        ),
    )
    args = parser.parse_args()

    report = {
        "protocol_note": (
            "Train-val curves only. Formal claim requires locked test: "
            "π0=frozen DAgger, π_best, π_final, MPC under identical seeds."
        ),
        "a6": None,
        "mappo": None,
    }
    if args.a6_dir.is_dir():
        report["a6"] = _matd3_summary(args.a6_dir)
        report["a6"]["interpretation"] = _interpret(report["a6"])
        report["a6"]["has_best_pt"] = (args.a6_dir / "best_capture.pt").is_file()
        report["a6"]["has_final_pt"] = (args.a6_dir / "final.pt").is_file()
        report["a6"]["has_initial_pt"] = (args.a6_dir / "initial.pt").is_file()
    if args.mappo_dir.is_dir():
        report["mappo"] = _mappo_summary(args.mappo_dir)
        report["mappo"]["interpretation"] = _interpret(report["mappo"])
        report["mappo"]["has_best_pt"] = (args.mappo_dir / "best_capture.pt").is_file()
        report["mappo"]["has_final_pt"] = (args.mappo_dir / "final.pt").is_file()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== O2O diagnosis (train-val) ===")
    for name in ("a6", "mappo"):
        block = report.get(name)
        if not block:
            print(f"{name}: missing")
            continue
        print(
            f"{name}: π0={block.get('pi0_capture')}@{block.get('pi0_episode')}  "
            f"best={block.get('best_capture')}@{block.get('best_episode')}  "
            f"final={block.get('final_capture')}@{block.get('final_episode')}"
        )
        for note in block.get("interpretation") or []:
            print(f"  - {note}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
