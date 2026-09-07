# Reproducible Experiment Guide

All figures and labels produced by the supplied plotting code are in English and use Times New Roman. Run commands from the project root:

```powershell
Set-Location D:\feikong\xieton
```

Create a Python 3.10--3.12 environment and install the declared dependencies before training:

```powershell
python -m pip install -r .\requirements.txt
python -c "import numpy, torch; print(torch.__version__, torch.cuda.is_available())"
```

## 1. Experimental protocol

Use five independent training seeds (`11,23,41,59,83`). Do not select a seed because it gives a favorable result. Evaluate every saved model on the same ten held-out scenario seeds (`101,113,127,139,151,163,179,191,211,227`). The heuristic has no trainable parameters and therefore requires evaluation only.

Keep the environment, reward, network width, training budget, and test seeds fixed when comparing MAPPO with QMIX. For an ablation, change exactly one switch relative to the full model. Use deterministic argmax actions during evaluation and never retrain a model during sensitivity analysis.

Before the final large run, use 200--500 episodes to verify the pipeline. For the paper, start with 2,000 episodes and extend to 5,000 or more only if the five-seed mean curve has not reached a stable plateau. Report the chosen stopping rule rather than choosing the best individual episode after viewing the test set.

## 2. Train the full MAPPO and QMIX models

Dry-run all commands first:

```powershell
python .\repro\run_paper_experiments.py --stage main --episodes 2000 --dry-run
```

Train both algorithms:

```powershell
python .\repro\run_paper_experiments.py --stage main --episodes 2000 --device auto
```

Resume interrupted runs:

```powershell
python .\repro\run_paper_experiments.py --stage main --episodes 2000 --device auto --resume-existing
```

The full checkpoints are written under:

```text
outputs/paper_experiments/mappo/full/
outputs/paper_experiments/qmix/full/
```

Each directory contains a final checkpoint for every seed (`mappo_seed_11.pt`, for example), an automatically updated `*_latest.pt`, per-seed CSV files, and a five-seed `*_mean_std.csv`.

## 3. Heuristic baseline

The heuristic policy is `nominal_search_policy` in `cooperative_search_env.py`. It is deterministic conditional on the environment state and has no optimization loop. Do not call a training command for it. Include it automatically by running `evaluate_policy_suite.py`.

## 4. MAPPO and QMIX ablations

Train the complete single-factor ablation suite:

```powershell
python .\repro\run_paper_experiments.py --stage ablations --episodes 2000 --device auto
```

Available variants are:

| Variant | Changed component | Scientific question |
|---|---|---|
| `no_gat` | Replaces the complete graph/heterogeneous encoder with a plain per-UAV MLP over the flat local observation | Does the graph encoder improve over a conventional MLP policy? |
| `no_heterogeneous_entities` | Removes explicit target/obstacle entity nodes | Does heterogeneous scene encoding help? |
| `no_revisit_map` | Removes revisit/staleness memory and neutrally masks all revisit-derived policy features while retaining the probability/uncertainty frontier | Does temporal coverage memory help? |
| `no_freshness_fusion` | Ignores belief age and link confidence in fusion | Does freshness-aware fusion help under outages? |
| `no_communication` | Removes all peer communication edges and peer-map fusion | Does the policy obtain measurable value from communication? |
| `full_communication` | Uses an always-connected communication graph | How far is the weak-communication model from an ideal communication upper bound? |
| `no_safety_filter` | Sets avoidance mode to `none` | What is the safety layer's contribution? |

Train a selected subset:

```powershell
python .\repro\run_paper_experiments.py --stage ablations `
  --variants no_gat,no_revisit_map,no_freshness_fusion,no_safety_filter `
  --episodes 2000
```

Use the same variants for both MAPPO and QMIX. Treat `full_communication` as an ideal-information upper bound rather than a strict component ablation. Under the default `team_potential` reward, there is no direct revisit reward, so a `no_revisit_reward` run would be identical to the full reward definition and is intentionally excluded.

## 5. Paired fixed-policy evaluation

The simplest option after all training runs finish is the automatic analysis entry point:

```powershell
python .\repro\run_paper_analysis.py --stage all
```

It automatically discovers the five checkpoints for MAPPO, QMIX, and every ablation; evaluates the heuristic on the same scenarios; performs sensitivity analysis; and writes the paper figures. The individual commands below remain useful when only selected experiments are required.

The evaluator accepts repeated labels, so all five independently trained models can be pooled under one method label. A complete main comparison is:

```powershell
python .\repro\evaluate_policy_suite.py `
  --policy MAPPO=.\outputs\paper_experiments\mappo\full\mappo_seed_11.pt `
  --policy MAPPO=.\outputs\paper_experiments\mappo\full\mappo_seed_23.pt `
  --policy MAPPO=.\outputs\paper_experiments\mappo\full\mappo_seed_41.pt `
  --policy MAPPO=.\outputs\paper_experiments\mappo\full\mappo_seed_59.pt `
  --policy MAPPO=.\outputs\paper_experiments\mappo\full\mappo_seed_83.pt `
  --policy QMIX=.\outputs\paper_experiments\qmix\full\qmix_seed_11.pt `
  --policy QMIX=.\outputs\paper_experiments\qmix\full\qmix_seed_23.pt `
  --policy QMIX=.\outputs\paper_experiments\qmix\full\qmix_seed_41.pt `
  --policy QMIX=.\outputs\paper_experiments\qmix\full\qmix_seed_59.pt `
  --policy QMIX=.\outputs\paper_experiments\qmix\full\qmix_seed_83.pt `
  --out-dir .\outputs\paper_evaluation\main
```

For an ablation table, repeat the command with labels such as `MAPPO-Full`, `MAPPO-No-GAT`, and `MAPPO-No-Freshness`. Include all five training seeds for each label. Use the same held-out scenario seeds for every label.

Primary effectiveness metrics should be target discovery rate, discovery AUC, and censored mean first-detection time (MFDT). Use coverage rate as a search-efficiency measure and collision rate plus safety-intervention rate as safety measures. Censor undiscovered targets at `horizon + 1`; averaging detection time only over successfully found targets is biased.

## 6. Sensitivity analysis

Use a fixed representative checkpoint from each algorithm (normally seed 23, chosen before inspecting sensitivity results):

```powershell
python .\repro\evaluate_checkpoint_sensitivity.py `
  --mappo .\outputs\paper_experiments\mappo\full\mappo_seed_23.pt `
  --qmix .\outputs\paper_experiments\qmix\full\qmix_seed_23.pt `
  --sweeps target_speed,obstacle_speed,comm_radius,sensor_radius,n_obstacles,outage_base_prob,outage_jammed_prob,outage_recovery_prob `
  --out-dir .\outputs\paper_evaluation\sensitivity
```

Use `--quick` only for a smoke test. Sensitivity is a fixed-policy robustness test: only the named environment parameter changes, and no fine-tuning occurs. The default values are included inside each sweep and should be visually identifiable as the reference operating point.

Recommended ranges:

- Target speed: `0.00, 0.25, 0.50, 0.75, 1.00`
- Obstacle speed: `0.00, 0.25, 0.50, 0.75`
- Communication radius: `4, 7, 10, 14`
- Sensor radius: `1.0, 1.5, 2.0, 2.5, 3.0`
- Obstacle count: `0, 5, 10, 20, 30`
- Base outage probability: `0.00, 0.02, 0.05, 0.10`
- Jammed outage probability: `0.10, 0.30, 0.45, 0.60`
- Outage recovery probability: `0.20, 0.35, 0.45, 0.60, 0.80`

Do not sweep the number of UAVs with the current QMIX checkpoint because the mixing network dimension depends on team size. A team-size generalization study requires separately trained checkpoints or a permutation-invariant mixer.

## 7. Paper figures

Main comparison:

```powershell
python .\repro\plot_paper_results.py --mode comparison `
  --input .\outputs\paper_evaluation\main\policy_suite_summary.csv `
  --output .\outputs\paper_figures\main_comparison.svg `
  --title "Performance Comparison under Weak Communication"
```

Ablation comparison:

```powershell
python .\repro\plot_paper_results.py --mode ablation `
  --input .\outputs\paper_evaluation\mappo_ablation\policy_suite_summary.csv `
  --output .\outputs\paper_figures\mappo_ablation.svg `
  --title "MAPPO Ablation Study"
```

Sensitivity figure:

```powershell
python .\repro\plot_paper_results.py --mode sensitivity `
  --input .\outputs\paper_evaluation\sensitivity\checkpoint_sensitivity_summary.csv `
  --parameter comm_radius `
  --output .\outputs\paper_figures\sensitivity_comm_radius.svg `
  --title "Sensitivity to Communication Radius"
```

Five-seed convergence:

```powershell
python .\repro\plot_paper_results.py --mode training `
  --input .\outputs\paper_experiments\mappo\full\mappo_mean_std.csv `
  --output .\outputs\paper_figures\mappo_convergence.svg `
  --title "MAPPO Training Convergence"
```

SVG is vector-based and suitable for direct insertion or conversion to PDF/EPS. Keep text editable during manuscript preparation. If the publisher requires raster figures, export at 600 dpi without changing the aspect ratio.

## 8. Reporting checklist

- Report five training seeds and ten held-out scenario seeds.
- State whether intervals are across models, scenarios, or all model-scenario runs.
- Use identical test scenarios for all methods.
- Report mean, standard deviation, and 95% confidence interval.
- Include absolute metrics and paired differences versus the heuristic.
- Keep reward out of the primary performance table; reward is an optimization signal, not the task objective.
- Never replace missing measurements with idealized values in the final paper.
