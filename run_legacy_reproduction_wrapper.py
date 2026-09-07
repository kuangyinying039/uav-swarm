"""Backward-compatible wrapper for the renamed heuristic demo entrypoint.

Use `run_reference_heuristic_demos.py` for reference mechanism demos. MARL
experiments live in `train_cooperative_marl.py` and
`train_multi_seed_cooperative_marl.py`.
"""

from __future__ import annotations

from run_reference_heuristic_demos import main


if __name__ == "__main__":
    main()
