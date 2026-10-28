# Frozen eval summaries

Compact capture-rate tables extracted from held-out Isaac evals (100 episodes/level, seeds from 7000000).

| File | Tag | Role |
|------|-----|------|
| `hybrid_postlc_eval_summary.json` | `isaac_dual_postlc` | Hybrid baseline only |
| `physx_closed_eval_summary.json` | `isaac_dual_physx_closed` | Formal pure-PhysX table (**target_diameter=0.75** archive) |
| `physx_closed_d05_baseline_eval_summary.json` | `isaac_dual_physx_closed` | Current ladder baselines (**target_diameter=0.5**) |

**Do not mix diameter=0.75 and diameter=0.5 tables.** See [`../TECHNICAL_SPEC.md`](../TECHNICAL_SPEC.md) and [`../SIM2REAL.md`](../SIM2REAL.md).

## Current d=0.5 hard snapshot (as committed)

| Method | hard capture |
|--------|--------------|
| APF | 0.04 |
| FRPN | 0.46 |
| MPC | 0.53 |
| π0 (DAgger frozen) | 0.47 |

Raw per-level JSONs live under `/srv/data/projects/kyy/runs/isaac_dual_physx_closed/train/isaac_dual_eval_baselines_d05/` (not in git).
