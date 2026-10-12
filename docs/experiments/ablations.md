# Ablations EXP-PROP-002…005 — one-component changes vs the main proposal

Each ablation changes exactly **one component** of EXP-PROP-001 (mlp_deepset);
everything else is identical: encoder 45→128→128, per-epoch HC latent
normative bank, comparator mlp on `[z, μ, z−μ, z·μ]` → 64, hyperparameters
epochs 150 / patience 30 / AdamW 1e-3 / wd 1e-4 / batch 16 / dropout 0.3 /
threshold 0.5.

Code: `src/models/proposal.py` with the flags below; config in
`src/registry.py::PROPOSAL_EXPERIMENTS`. Artifacts:
`outputs/experiments/EXP-PROP-00X/full45/seed<seed>/`; per-seed fold-mean
AUCs in `outputs/experiments/reports/EXP-PROP-00X.md`.

## Overview (10 seeds, official 4-fold)

| EXP | Component changed | AUC | ΔAUC vs 001 (pp) |
|---|---|---|---|
| EXP-PROP-001 (mlp_deepset) | — (main proposal) | 0.9403 ± 0.0156 | — |
| EXP-PROP-002 (mlp_mean) | pooling: mean | 0.9383 ± 0.0064 | -0.20 |
| EXP-PROP-003 (mlp_attn) | pooling: attention | 0.9299 ± 0.0077 | -1.04 |
| EXP-PROP-004 (z_mean) | deviation: fixed hard z-score | 0.9073 ± 0.0040 | -3.30 |
| EXP-PROP-005 (mlp_max) | pooling: masked max | 0.9304 ± 0.0156 | -0.99 |

ΔAUC = mean-of-means difference (paired per-seed deltas: −0.0020, −0.0104,
−0.0330, −0.0099). ACC/SEN/SPEC/F1 in
[`EXPERIMENTS_RESULTS.md`](../../EXPERIMENTS_RESULTS.md). Development
numbers over the same 10 seeds/folds — not significance tests.

## EXP-PROP-002 (mlp_mean) — mean pooling

- **Delta**: `d_mean = Σ(mask·d) / count(valid stimuli)` → 64-dim subject
  embedding; head input 64 (no max path).
- **Tests**: whether the max path of the deepset captures extreme/atypical
  stimuli that the mean alone misses.
- **Result**: AUC 0.9383 ± 0.0064 | pooled OOF AUC 0.8957.
- **Reading**: the smallest gap among the pooling ablations (−0.20 pp) —
  the average deviation across stimuli already summarizes the subject well.

## EXP-PROP-003 (mlp_attn) — attention pooling

- **Delta**: learned scorer 64 → 32 → ReLU → 1; attention
  `a_s = softmax(scorer(d_s) + (mask_s − 1)·1e9)`; subject embedding =
  `Σ a_s·mask_s·d_s` (64-d). Missing stimuli get −∞ logits, never weighted.
- **Tests**: whether learned, data-dependent stimulus weighting beats the
  fixed mean‖max combination.
- **Result**: AUC 0.9299 ± 0.0077 | pooled OOF AUC 0.8790.
- **Reading**: clearly below deepset (−1.04 pp). With ~120 train subjects
  the scorer adds parameters that the fixed combination already covers.

## EXP-PROP-004 (z_mean) — fixed hard z-deviation

- **Delta**: the learned deviation machinery is replaced by per-stimulus
  z-scores `z = (x_sf − μ_sf) / σ_sf` where μ, σ are per-stimulus mean/std
  over **train-fold HC subjects only**, computed in raw feature space (no
  encoder, no comparator, no latent bank — the model has no parameters
  before the head). Missing feature cells filled with μ (zero deviation).
  Mean pooling → 45-dim embedding; head 45 → 64 → BN → ReLU → Dropout(0.3) → 1.
- **Tests**: how much of the main proposal's value comes from the *learned*
  deviation versus the classic fixed normative z-deviation; pooling and
  head are shared, so the comparison isolates the deviation mechanism.
- **Result**: AUC 0.9073 ± 0.0040 | pooled OOF AUC 0.8966.
- **Reading**: the largest gap in the suite (−3.30 pp) — the learned
  pipeline (latent deviation + mlp comparator + bank) adds substantial
  value over raw feature-space z-scores; the hard z-deviation alone sits
  below fnn (0.9176) and barely above svm_rbf (0.8793). Small seed SD is
  expected: the deviation path is fixed, only the head trains.

## EXP-PROP-005 (mlp_max) — masked max pooling

- **Delta**: `d_max = max over valid stimuli of (mask·d + (1−mask)·(−1e9))`
  → 64-dim embedding; head input 64 (no mean path).
- **Tests**: the max-only half of the deepset — whether the most extreme
  per-stimulus deviation per dimension (the "worst stimulus" view of
  normative atypicality) alone carries the subject-level signal.
- **Result**: AUC 0.9304 ± 0.0156 | pooled OOF AUC 0.9044.
- **Reading**: below deepset but above attention (−0.99 pp). Pooling
  ranking on mean AUC: deepset (0.9403) > mean (0.9383) > max (0.9304) >
  attention (0.9299) — the mean‖max combination carries the strongest
  signal.
