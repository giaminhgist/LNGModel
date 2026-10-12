# EMS — Learned Normative Gaze Modeling for Schizophrenia Recognition

Research codebase for **stimulus-conditioned learned normative deviations**
of eye-tracking behavior for schizophrenia (SZ) recognition, on the public
**EMS** dataset (Song et al., *IEEE TNNLS* 2024 —
[paper](https://ieeexplore.ieee.org/document/10645682)).

The main proposal (EXP-PROP-001) learns, per stimulus, a latent
representation of fixation behavior, compares each subject against a
**healthy-control (HC) normative bank** recomputed from the training fold,
and pools the per-stimulus deviations into a subject embedding classified
by an MLP head. On the official 4-fold protocol with 10 locked seeds it
reaches **AUC 0.9403 ± 0.0156** (validation), above all 10 baselines and
the published MSNet reference (AUC 0.8972).

- **Results**: [`EXPERIMENTS_RESULTS.md`](EXPERIMENTS_RESULTS.md)
- **Documentation**: [`docs/`](docs/) — methodology and results of
  every experiment + the full 45-feature reference

## Table of contents

1. [Project summary](#1-project-summary)
2. [Dataset](#2-dataset)
3. [The 45 hand-crafted features](#3-the-45-hand-crafted-features)
4. [Research question, hypothesis, novelty](#4-research-question-hypothesis-novelty)
5. [Main results — proposal vs baselines vs MSNet](#5-main-results--proposal-vs-baselines-vs-msnet)
6. [Ablation studies](#6-ablation-studies)
7. [Reproducing all experiments](#7-reproducing-all-experiments)
8. [Repository layout](#8-repository-layout)
9. [Outputs](#9-outputs)
10. [Data access & licensing](#10-data-access--licensing)
11. [Documentation map](#11-documentation-map)

## 1. Project summary

- **Task**: subject-level SZ vs HC classification from free-viewing
  eye-tracking (fixations) — 45 hand-crafted features per (subject,
  stimulus), official 4-fold validation.
- **Method (EXP-PROP-001, main proposal)**: encoder 45→128→128 per
  stimulus → per-epoch HC latent normative bank (per-stimulus mean/std over
  train-fold HC encodings) → learned comparator on `[z, μ, z−μ, z·μ]` →
  per-stimulus deviation (64-d) → **deepset pooling** (mean‖max) →
  subject embedding → MLP head. Details: [`docs/experiments/EXP-PROP-001.md`](docs/experiments/EXP-PROP-001.md).
- **Baselines**: 10 classical-ML/FNN baselines on three subject-level
  representations (agg 91-d, catagg 181-d, concat 4500-d).
  Details: [`docs/experiments/baselines.md`](docs/experiments/baselines.md).
- **Ablations**: EXP-PROP-002…005 isolate pooling (mean / attention / max)
  and the deviation mechanism (fixed hard z-deviation).
  Details: [`docs/experiments/ablations.md`](docs/experiments/ablations.md).
- **Protocol**: official subject-level 4-fold (Set_0..Set_3, 120 train /
  40 val per fold), 10 locked seeds `[1234, 100, 0, 2024, 42, 3, 123, 13,
  64, 2026]`, best checkpoint = highest outer-validation AUC (earliest
  epoch wins ties), threshold 0.5, SZ = 1. CPU-only
  (torch 2.11.0+cu128, sklearn 1.9.0).

## 2. Dataset

**EMS** (Song et al., IEEE TNNLS 2024) — public free-viewing eye-tracking
benchmark:

- 160 labelled subjects: 80 HC (ids < 200) / 80 SZ (ids ≥ 200), each
  viewing 100 images (natural, social, synthetic, manipulated); 48 official
  test subjects with **withheld labels** (all reported metrics are
  validation metrics).
- Fixation tables per subject: `IMAGE, FIX_INDEX, FIX_DURATION, FIX_X,
  FIX_Y, FIX_PUPIL` on a 1024×768 display.
- Cleaning (`src/scripts/preprocess.py::clean_fixations`): drop off-screen
  fixations, keep duration in (40, 2000] ms, drop per-stimulus pupil
  outliers (±4 SD of the stimulus median); stimuli with < 2 clean
  fixations are missing (masked rows).
- Preprocessing writes the feature table to `processed_dataset/`
  (MultiIndex `(subject_id, image)`, NaN for missing pairs).

## 3. The 45 hand-crafted features

One 45-dim vector per (subject, stimulus), computed from cleaned fixations
ordered by `FIX_INDEX` (`src/features/stimulus.py`):

| group | count | examples |
|---|---|---|
| spatial — position & dispersion (`spa_`) | 17 | `spa_fix_count`, `spa_std_x/y`, `spa_dispersion`, `spa_bbox_area`, center bias (`spa_center_dist_*`, `spa_center_frac`), quadrant fractions (`spa_q1..q4`), 8×6-grid `spa_entropy` / `spa_max_grid_frac`, `spa_skew_x` |
| scanpath geometry (`geo_`) | 10 | `geo_scanpath_len`, `geo_sacc_amp_mean/std/max`, `geo_dx/dy_mean`, `geo_angle_var`, `geo_revisit_rate`, `geo_nn_dist_mean`, `geo_hull_area` |
| temporal (`tem_`) | 11 | `tem_dur_mean/std/total/max`, `tem_first/last_dur`, `tem_ifi_mean/std`, `tem_velocity_mean`, `tem_fix_rate`, `tem_trans_entropy` |
| pupil (`pup_`) | 7 | `pup_mean/std/min/max/median`, `pup_slope`, `pup_first_last_diff` |

Full reference — definition, exact formula, units, missingness rules, and
behavioral/clinical interpretation for every feature:
[`docs/features45.md`](docs/features45.md).

## 4. Research question, hypothesis, novelty

- **Research question**: can stimulus-conditioned *normative deviations* of
  eye-movement behavior — learned jointly with a classifier from
  free-viewing fixation features — recognize schizophrenia more accurately
  than classical and neural baselines on aggregated hand-crafted features?
- **Hypothesis**: a subject's per-stimulus atypicality relative to
  HC stimulus-specific norms carries discriminative SZ/HC signal beyond
  subject-level aggregates of the 45 features, and a learned deviation
  (latent encoder + HC bank + comparator) captures this better than fixed
  z-scores in raw feature space.
- **Novelty / contribution** (engineering-scientific, on a public
  benchmark): a trainable normative-deviation pipeline — HC latent bank
  recomputed every epoch from the training fold, comparator on
  `[z, μ, z−μ, z·μ]`, masked deepset pooling — evaluated under a strict
  leakage-safe protocol. It outperforms all included baselines and the
  published MSNet validation AUC; the ablations (002–005) attribute the
  gain to the learned deviation mechanism and the deepset pooling.

## 5. Main results — proposal vs baselines vs MSNet

10 seeds, official 4-fold; per seed = mean of the 4 fold metrics, reported
as mean ± sample SD (ddof=1) over the seed means (full tables and
comparisons in [`EXPERIMENTS_RESULTS.md`](EXPERIMENTS_RESULTS.md)).

| Method | AUC | ACC | SEN | SPEC | F1 | ΔAUC vs main (pp) |
|---|---|---|---|---|---|---|
| **EXP-PROP-001 (mlp_deepset, main)** | **0.9403 ± 0.0156** | **0.8494** | 0.8229 | **0.8717** | **0.8380** | — |
| fnn | 0.9176 ± 0.0082 | 0.8344 | **0.8419** | 0.8263 | 0.8329 | -2.28 |
| fnn_cat | 0.9049 ± 0.0059 | 0.8125 | 0.7819 | 0.8417 | 0.7989 | -3.54 |
| svm_rbf | 0.8793 ± 0.0000 | 0.8012 | 0.8168 | 0.7819 | 0.8032 | -6.10 |
| lr | 0.8711 ± 0.0000 | 0.8063 | 0.8134 | 0.8000 | 0.8074 | -6.93 |
| rf | 0.8581 ± 0.0038 | 0.8063 | 0.7854 | 0.8191 | 0.7993 | -8.22 |
| qda | 0.8229 ± 0.0000 | 0.7125 | 0.7494 | 0.6847 | 0.7225 | -11.75 |
| svm_linear | 0.8189 ± 0.0001 | 0.7469 | 0.7473 | 0.7501 | 0.7463 | -12.14 |
| knn | 0.8120 ± 0.0000 | 0.7312 | 0.6892 | 0.7729 | 0.7078 | -12.83 |
| gnb | 0.7913 ± 0.0000 | 0.7125 | 0.6264 | 0.7812 | 0.6756 | -14.90 |
| lr_l1 | 0.7360 ± 0.0035 | 0.6863 | 0.6982 | 0.6760 | 0.6797 | -20.43 |

**vs MSNet** (Song et al., TNNLS 2024 — published numbers, NOT retrained;
unpaired comparison, no superiority claim; MSNet uses an Otsu threshold per
validation set for ACC/SEN/SPEC/F1 while we use a fixed 0.5 threshold):
MSNet AUC 0.8972 → main proposal **+4.31 pp AUC**; all 5 proposal variants
and both FNN baselines sit above the published MSNet validation AUC.

## 6. Ablation studies

Each ablation changes exactly one component of the main proposal
(details: `docs/experiments/ablations.md`):

| EXP | Component changed | AUC | ACC | SEN | SPEC | F1 | ΔAUC vs main (pp) |
|---|---|---|---|---|---|---|---|
| EXP-PROP-002 (mlp_mean) | pooling: mean vs deepset | 0.9383 ± 0.0064 | 0.8375 | 0.8224 | 0.8480 | 0.8271 | -0.20 |
| EXP-PROP-003 (mlp_attn) | pooling: attention vs deepset | 0.9299 ± 0.0077 | 0.8219 | 0.8028 | 0.8399 | 0.8080 | -1.04 |
| EXP-PROP-004 (z_mean) | deviation: fixed hard z-deviation (no encoder/comparator/bank) | 0.9073 ± 0.0040 | 0.8094 | 0.8165 | 0.8058 | 0.8048 | -3.30 |
| EXP-PROP-005 (mlp_max) | pooling: masked max vs deepset | 0.9304 ± 0.0156 | 0.8319 | 0.8103 | 0.8534 | 0.8185 | -0.99 |

ΔAUC column = mean-of-means difference vs EXP-PROP-001 (derived from the
table; paired per-seed deltas are in `EXPERIMENTS_RESULTS.md`).

Reading: deepset (mean‖max) carries the strongest pooling signal; the
learned deviation pipeline adds substantial value over raw feature-space
z-scores (−3.30 pp when removed). Development numbers over the same 10
seeds/folds — not significance tests.

## 7. Reproducing all experiments

```bash
uv sync                     # Python ≥ 3.12 (see pyproject.toml)

# 0. data preparation (raw EMS must be in original_dataset/EMS/)
python src/scripts/preprocess.py

# 1. one proposal run (4 official folds, one seed)
python src/scripts/train_proposal.py --experiment EXP-PROP-001 --ablation mlp_deepset \
    --comparator mlp --pool deepset --fold all --seed 42
python src/scripts/train_proposal.py --experiment EXP-PROP-002 --ablation mlp_mean \
    --comparator mlp --pool mean --fold all --seed 42
python src/scripts/train_proposal.py --experiment EXP-PROP-003 --ablation mlp_attn \
    --comparator mlp --pool attention --fold all --seed 42
python src/scripts/train_proposal.py --experiment EXP-PROP-004 --ablation z_mean \
    --deviation z --pool mean --fold all --seed 42
python src/scripts/train_proposal.py --experiment EXP-PROP-005 --ablation mlp_max \
    --comparator mlp --pool max --fold all --seed 42

# 2. one baseline
python src/scripts/run_baseline.py --method svm_rbf --seed 42

# 3. full 10-seed matrix (5 proposal + 10 baselines, official 4-fold;
#    single-threaded BLAS env is set by the runner)
python src/scripts/run_experiments.py --workers 8

# 4. aggregation + reports (outputs/experiments/*.csv, EXPERIMENTS_RESULTS.md)
python src/scripts/summarize_experiments.py

# 5. seed/correctness checks
python src/checks/check_seed.py
python src/checks/check_correctness.py
```

Notes: the experiment runner is resume-safe (a job is skipped only when its
artifacts are complete AND the recorded data/code hashes match); every
run is deterministic per seed/fold/device; locked hyperparameters live in
`src/registry.py` (epochs 150, patience 30, AdamW lr 1e-3, wd 1e-4,
batch 16, dropout 0.3 for the proposal). Runs recorded before the 2026-10-05
`src/` restructure carry the pre-restructure `code_hash`, so the runner
treats them as `stale` and re-runs them (identical metrics, locked seeds) to
refresh the recorded hash.

## 8. Repository layout

```
src/
├── config.py            # shared paths + constants (ROOT/RAW/PROCESSED/OUTPUTS, screen, categories)
├── rawdata.py           # raw EMS xlsx loading, subject labels, xlsx-based official folds
├── data/                # processed-data access, metadata-based folds, subject matrices, hashes
├── features/            # stimulus.py (45 features) + subject.py (baseline subject-level reps)
├── models/              # proposal.py (normative model) + baselines.py (classical ML + FNN)
├── training/            # generic seeded trainer: config, epoch logs, checkpoints, bank hook
├── scripts/             # entry points: preprocess, train_proposal, run_baseline,
│                        #   run_experiments (resume-safe), summarize_experiments
├── metrics.py, seeding.py  # shared metric implementation + single seeding source + code hash
├── registry.py          # locked SEEDS + EXP-PROP-001..005 + baselines
└── checks/              # seed/correctness verification suite
docs/                                      # documentation (see map below)
outputs/experiments/                       # all run artifacts (per experiment/seed/fold)
processed_dataset/                         # generated feature tables + metadata
```

## 9. Outputs

Every run writes to `outputs/experiments/{experiment}/{feature_set}/seed{seed}/fold{fold}/`
(config, run info with code/data hashes, per-epoch `metrics.jsonl`,
per-epoch val probabilities, `best.pt`/`last.pt` including normative-bank
buffers, predictions, summary). Aggregates: `seed_metrics.csv`,
`fold_metrics.csv`, `summary.csv`, `reports/*.md`.

## 10. Data access & licensing

The EMS dataset is **not** included in this repository (usage agreement:
non-commercial research only). Download it from the
[official repo](https://github.com/YingjieSong1/EMS) into
`original_dataset/EMS/`, then regenerate `processed_dataset/` with
`python src/scripts/preprocess.py`.

## 11. Documentation map

| file | content |
|---|---|
| [`EXPERIMENTS_RESULTS.md`](EXPERIMENTS_RESULTS.md) | authoritative result tables (main vs baselines, ablations, MSNet) |
| [`docs/features45.md`](docs/features45.md) | the 45 features: definition, formula, units, interpretation |
| [`docs/experiments/EXP-PROP-001.md`](docs/experiments/EXP-PROP-001.md) | main proposal — learned normative deviation + deepset pooling |
| [`docs/experiments/ablations.md`](docs/experiments/ablations.md) | EXP-PROP-002…005 — one-component ablations (pooling, deviation) |
| [`docs/experiments/baselines.md`](docs/experiments/baselines.md) | the 10 classical-ML / FNN baselines |

Numbers in `docs/` are copied from `EXPERIMENTS_RESULTS.md` (root) and
`outputs/experiments/reports/*.md`; the per-run artifacts live under
`outputs/experiments/`. The executable code in `src/` is the ground truth
whenever a formula in the docs looks ambiguous.
