# EXP-PROP-001 — main proposal (mlp_deepset)

**Learned stimulus-conditioned normative modeling.** The main proposal of
the suite: a learned encoder maps each stimulus feature vector to a latent
z; a HC latent normative bank (per-stimulus mean/variance over the
train-fold HC encodings) is recomputed every epoch; a learned comparator
produces per-stimulus deviation d; a set-level pool aggregates the 100
deviations into one subject embedding; an MLP head classifies HC/SZ.

Code: `src/models/proposal.py` (`NormativeModel`, `NormativeDataset`),
training in `src/scripts/train_proposal.py` + `src/training/trainer.py`.
Artifacts: `outputs/experiments/EXP-PROP-001/full45/seed<seed>/`.

## Architecture (input dim 45)

1. **Encoder** `f_θ`: 45 → Linear 128 → LayerNorm → GELU → Linear 128
   (latent z per stimulus).
2. **HC latent normative bank**: buffers `bank_mu`/`bank_sigma`
   (100 stimuli × 128). Refreshed once per epoch from the train-fold HC
   encodings: per-stimulus mean and `sqrt(var).clamp(EPS) + EPS`
   (EPS = 1e-6). Schedule is lagged by one epoch so the best checkpoint's
   state contains exactly the bank its validation metrics were computed
   with. Uses HC subjects only, exactly as specified by the normative
   design (no diagnosis labels in the bank statistics themselves).
3. **Comparator** `g_φ` (mlp): on `[z, μ, z−μ, z·μ]` (512 dims) →
   256 → ReLU → 128 → ReLU → 64 deviation vector d per stimulus.
4. **Deepset pooling**: `mean(d) ‖ max(d)` over the valid stimuli
   (masked) → 128-dim subject embedding.
5. **Head**: 128 → 64 → BatchNorm → ReLU → Dropout(0.3) → 1 → sigmoid =
   P(SZ).

**Inputs/preprocessing** (leakage-safe, `src/data/tabular.py`): raw 45
features standardized per-feature with mean/std fitted on the training-fold
subjects only; missing feature cells filled with the train-fitted
per-feature mean (zero deviation); missing stimuli (all-NaN rows) masked
out of pooling, bank and losses.

**Hyperparameters** (locked, identical across the suite): epochs 150,
patience 30 (early stop on outer-val AUC, earliest epoch wins ties), AdamW
lr 1e-3, wd 1e-4, batch 16, dropout 0.3, threshold 0.5, SZ = 1.

## What it tests

The full learned normative pipeline: latent deviation + learned comparator +
deepset pooling. It is the reference for all ablations (002–005) and the
main method compared against the 10 baselines and the published MSNet.

## Results (10 seeds, official 4-fold)

- AUC **0.9403 ± 0.0156** | ACC 0.8494 | SEN 0.8229 | SPEC 0.8717 |
  F1 0.8380 | pooled OOF AUC 0.9179.
- Per-seed fold-mean AUCs: s0=0.952, s3=0.939, s13=0.928, s42=0.928,
  s64=0.926, s100=0.962, s123=0.931, s1234=0.967, s2024=0.948, s2026=0.924.

vs the published MSNet reference (AUC 0.8972): +4.31 pp (unpaired
comparison against published numbers — no statistical superiority claim).

## Interpretation and limitations

- Beats all 10 baselines on mean AUC (best baseline fnn: 0.9176) and all
  5 proposal variants beat the published MSNet validation AUC.
- The gain over fnn (+2.28 pp) is the learned normative deviation +
  deepset pooling over a plain MLP on subject aggregates.
- Development numbers over the same 10 seeds/folds; MSNet was not retrained
  (published numbers, different threshold policy).
