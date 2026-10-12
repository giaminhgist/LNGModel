# EXPERIMENTS_RESULTS — EMS-Projects (10 seeds, official 4-fold)

Generated 2026-09-23 from `outputs/experiments/` by `src/scripts/summarize_experiments.py`.

## Protocol (brief)

- Official subject-level 4-fold validation (Set_0..3; 120 train / 40 val subjects per fold); same folds and same 10 seeds ([1234, 100, 0, 2024, 42, 3, 123, 13, 64, 2026]) for every experiment.
- Best checkpoint = highest outer-validation AUC (earliest epoch wins ties); ACC/SEN/SPEC/F1 at that checkpoint. Threshold 0.5, SZ=1. Development comparison with selection optimism — not an unbiased generalization estimate.
- Per seed = mean of the 4 fold metrics; reported as mean ± sample SD (ddof=1) over the 10 seed means. Classical ML: one fit per fold (no epochs); FNN: unified checkpoint policy (no inner split).
- CPU-only (torch 2.11.0+cu128, sklearn 1.9.0).

## 1. Main proposal vs baselines

The main proposal is **EXP-PROP-001 (mlp_deepset)** (learned latent normative deviation + learned mlp comparator + deepset pooling, 45 hand-crafted features).

| Method | AUC | ACC | SEN | SPEC | F1 | ΔAUC vs main (pp) |
|---|---|---|---|---|---|---|
| EXP-PROP-001 (mlp_deepset) | 0.9403 ± 0.0156 | 0.8494 | 0.8229 | 0.8717 | 0.8380 | — |
| svm_rbf | 0.8793 ± 0.0000 | 0.8012 | 0.8168 | 0.7819 | 0.8032 | -6.10 |
| svm_linear | 0.8189 ± 0.0001 | 0.7469 | 0.7473 | 0.7501 | 0.7463 | -12.14 |
| rf | 0.8581 ± 0.0038 | 0.8063 | 0.7854 | 0.8191 | 0.7993 | -8.22 |
| qda | 0.8229 ± 0.0000 | 0.7125 | 0.7494 | 0.6847 | 0.7225 | -11.75 |
| gnb | 0.7913 ± 0.0000 | 0.7125 | 0.6264 | 0.7812 | 0.6756 | -14.90 |
| lr | 0.8711 ± 0.0000 | 0.8063 | 0.8134 | 0.8000 | 0.8074 | -6.93 |
| knn | 0.8120 ± 0.0000 | 0.7312 | 0.6892 | 0.7729 | 0.7078 | -12.83 |
| fnn | 0.9176 ± 0.0082 | 0.8344 | 0.8419 | 0.8263 | 0.8329 | -2.28 |
| fnn_cat | 0.9049 ± 0.0059 | 0.8125 | 0.7819 | 0.8417 | 0.7989 | -3.54 |
| lr_l1 | 0.7360 ± 0.0035 | 0.6863 | 0.6982 | 0.6760 | 0.6797 | -20.43 |

- Strongest baseline: **fnn** (AUC 0.9176) — the main proposal is +2.28 pp above it; no baseline beats the main proposal on mean AUC.
- Deterministic baselines (gnb, knn, lr, qda) have seed SD 0 by construction — that is not evidence of stability.

## 2. Ablation studies (each EXP vs the main proposal)

| EXP | Component changed | AUC | ACC | SEN | SPEC | F1 |
|---|---|---|---|---|---|---|
| EXP-PROP-002 (mlp_mean) | pooling ablation: mean vs deepset (main) | 0.9383 ± 0.0064 | 0.8375 | 0.8224 | 0.8480 | 0.8271 |
| EXP-PROP-003 (mlp_attn) | pooling ablation: attention vs deepset (main) | 0.9299 ± 0.0077 | 0.8219 | 0.8028 | 0.8399 | 0.8080 |
| EXP-PROP-004 (z_mean) | deviation ablation: fixed hard z-deviation (no encoder/comparator/bank) vs learned deviation (main) | 0.9073 ± 0.0040 | 0.8094 | 0.8165 | 0.8058 | 0.8048 |
| EXP-PROP-005 (mlp_max) | pooling ablation: masked max vs deepset (main) | 0.9304 ± 0.0156 | 0.8319 | 0.8103 | 0.8534 | 0.8185 |

Reading the ablations:
- **Pooling**: mean (002), attention (003) and max (005) are all below deepset (main) on mean AUC — deepset (mean‖max) carries the strongest signal; the gap is smallest for mean.
- **Deviation mechanism**: the fixed hard z-deviation (004, no encoder/comparator/bank) trails the learned deviation (main) — the learned pipeline adds value over raw feature-space z-scores.
- Means are development numbers over the same 10 seeds and folds, not significance tests.

## 3. Proposal suite vs MSNet (published reference)

MSNet (Song et al., TNNLS 2024, 4-fold validation, per the official benchmark ReadMe): AUC 0.8972, ACC 0.8313, SEN 0.8051, SPEC 0.8708, F1 0.8244. MSNet was NOT retrained with our seeds — this is a comparison against published numbers (unpaired; no statistical superiority claim). MSNet's official workflow uses an Otsu threshold per validation set for ACC/SEN/SPEC/F1 while our table uses a fixed 0.5 threshold.

| Experiment | ΔAUC (pp) | ΔACC | ΔSEN | ΔSPEC | ΔF1 |
|---|---|---|---|---|---|
| EXP-PROP-001 (mlp_deepset) | +4.31 | +1.81 | +1.78 | +0.09 | +1.36 |
| EXP-PROP-002 (mlp_mean) | +4.11 | +0.62 | +1.73 | -2.28 | +0.27 |
| EXP-PROP-005 (mlp_max) | +3.32 | +0.06 | +0.52 | -1.74 | -0.59 |
| EXP-PROP-003 (mlp_attn) | +3.27 | -0.94 | -0.23 | -3.09 | -1.64 |
| fnn | +2.04 | +0.31 | +3.68 | -4.45 | +0.85 |
| EXP-PROP-004 (z_mean) | +1.01 | -2.19 | +1.14 | -6.50 | -1.96 |
| fnn_cat | +0.77 | -1.88 | -2.32 | -2.91 | -2.55 |
| svm_rbf | -1.79 | -3.01 | +1.17 | -8.89 | -2.12 |
| lr | -2.61 | -2.51 | +0.83 | -7.08 | -1.70 |
| rf | -3.91 | -2.51 | -1.97 | -5.17 | -2.51 |
| qda | -7.43 | -11.88 | -5.57 | -18.61 | -10.19 |
| svm_linear | -7.83 | -8.44 | -5.78 | -12.07 | -7.81 |
| knn | -8.52 | -10.01 | -11.59 | -9.79 | -11.66 |
| gnb | -10.59 | -11.88 | -17.87 | -8.96 | -14.88 |
| lr_l1 | -16.12 | -14.51 | -10.69 | -19.48 | -14.47 |

- All 5 proposal configs and both FNN baselines sit above the published MSNet validation AUC (7 rows above it in total); the best classical method (svm_rbf) is below.

