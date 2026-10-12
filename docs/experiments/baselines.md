# Baselines — classical ML and FNN (10 methods)

Code: representations in `src/features/subject.py`, estimators in
`src/models/baselines.py`, protocol in `src/data/common.py::official_folds`,
matrix config in `src/registry.py::BASELINE_METHODS`. Per-run artifacts:
`outputs/experiments/<method>/seed<seed>/fold<fold>/`.

## Representations (subject level)

The 45 features are per (subject, stimulus). Baselines aggregate them into
one vector per subject (see `src/features/subject.py`):

| representation | dim | definition |
|---|---|---|
| `agg` | 91 | mean + std of each of the 45 features over the subject's valid stimuli, plus `n_valid_stim` (a derived counter, NOT one of the 45) |
| `catagg` | 181 | mean of each feature within each of the 4 stimulus categories (social, natural, synthetic, manipulated) plus `n_valid_stim` |
| `concat` | 4500 | raw concatenation of the 100 per-stimulus vectors (fixed image order) |

Missing (subject, stimulus) pairs are NaN and skipped in the aggregations.

## Methods and hyperparameters

All sklearn pipelines: `SimpleImputer(median)` → `StandardScaler()` →
estimator, fitted on the training fold only (no leakage). One fit per fold
(no epochs). The run seed reaches every stochastic estimator as
`random_state`.

| method | representation | estimator / config |
|---|---|---|
| `svm_rbf` | agg | SVC(C=1, kernel='rbf', gamma='scale', probability=True) |
| `svm_linear` | agg | SVC(C=1, kernel='linear', probability=True) |
| `rf` | agg | RandomForest(n_estimators=500, n_jobs=1) |
| `qda` | agg | PCA(20) → QDA(reg_param=0.5) (91 features > samples per class) |
| `gnb` | agg | GaussianNB |
| `lr` | agg | LogisticRegression(C=1, max_iter=2000) |
| `knn` | agg | KNeighbors(n_neighbors=5) |
| `fnn` | agg | FNN 91→128→64→32 (BN/ReLU/Dropout 0.3)→1; HP epochs 150, patience 20, AdamW lr 1e-3, wd 1e-4, batch 16; best-val-AUC checkpoint |
| `fnn_cat` | catagg | same FNN (input 181) |
| `lr_l1` | concat | LogisticRegression(C=1, penalty='l1', solver='liblinear', max_iter=2000) |

## Results (10 seeds, official 4-fold; mean ± SD over seed means)

| method | AUC | ACC | SEN | SPEC | F1 | ΔAUC vs main (pp) |
|---|---|---|---|---|---|---|
| fnn | 0.9176 ± 0.0082 | 0.8344 | 0.8419 | 0.8263 | 0.8329 | -2.28 |
| svm_rbf | 0.8793 ± 0.0000 | 0.8012 | 0.8168 | 0.7819 | 0.8032 | -6.10 |
| lr | 0.8711 ± 0.0000 | 0.8063 | 0.8134 | 0.8000 | 0.8074 | -6.93 |
| rf | 0.8581 ± 0.0038 | 0.8063 | 0.7854 | 0.8191 | 0.7993 | -8.22 |
| svm_linear | 0.8189 ± 0.0001 | 0.7469 | 0.7473 | 0.7501 | 0.7463 | -12.14 |
| qda | 0.8229 ± 0.0000 | 0.7125 | 0.7494 | 0.6847 | 0.7225 | -11.75 |
| gnb | 0.7913 ± 0.0000 | 0.7125 | 0.6264 | 0.7812 | 0.6756 | -14.90 |
| knn | 0.8120 ± 0.0000 | 0.7312 | 0.6892 | 0.7729 | 0.7078 | -12.83 |
| fnn_cat | 0.9049 ± 0.0059 | 0.8125 | 0.7819 | 0.8417 | 0.7989 | -3.54 |
| lr_l1 | 0.7360 ± 0.0035 | 0.6863 | 0.6982 | 0.6760 | 0.6797 | -20.43 |

"main" = EXP-PROP-001 (AUC 0.9403 ± 0.0156). Deterministic methods (gnb,
knn, lr, qda, svm_*) have seed SD 0 by construction — that is not evidence
of stability.

## Reading

- The strongest baseline is **fnn** (0.9176); no baseline reaches the main
  proposal on mean AUC. svm_rbf is the best classical method.
- Classical linear methods (svm_linear, lr) and lr_l1 on the 4500-dim
  concatenation sit clearly below the non-linear methods — the
  subject-stimulus interaction structure needs non-linear pooling.
- All numbers are development validation numbers (same 10 seeds and folds),
  not external generalization estimates.

Per-seed fold-mean AUCs are in `outputs/experiments/reports/<method>.md`;
raw artifacts in `outputs/experiments/<method>/`.
