"""Tabular (45-dim hand-crafted feature) access for the proposal.

Provides per-subject stimulus matrices X (S x D) + presence masks, plus
leakage-safe normative statistics: everything is fitted ONLY on the given
train subjects (HC-only where stated), then applied to any other subjects.

Missingness rules (documented numerical correction, applied from the first
Phase-3 run):
  - A (subject, stimulus) row whose features are ALL NaN is a *missing
    stimulus*: it is masked out everywhere (pooling, bank, losses).
  - A cell that is NaN while its row is otherwise valid is a *missing
    feature*: it is filled with the train-fitted per-feature mean (the HC
    stimulus norm for the hard-deviation modes), so it contributes ZERO
    deviation. (The old code filled it with 0.0, which produced an arbitrary
    large negative deviation; this is the recorded correction.)
  - Zero-variance features: std is clamped to EPS=1e-6, so a constant feature
    contributes exactly (x - mu) / EPS = 0 deviation everywhere.
The stimulus mask is defined from the all-NaN rows BEFORE any feature
selection, so dropping a column can never turn a stimulus into "missing".
"""
import numpy as np
import pandas as pd

from data.common import (NUM_FEATURES, NUM_STIMULI, load_feature_names,
                         load_stimulus_features, image_list)

EPS = 1e-6


def subject_matrices(subject_ids, partition="train", feature_cols=None):
    """Return {subject_id: (X (S, D) float32, mask (S,) bool)}.

    X rows are NaN for missing (subject, stimulus) pairs or missing feature
    cells; mask marks rows that are not entirely NaN. D = len(feature_cols)
    (default: the 45-column processed schema).
    """
    feature_cols = list(feature_cols) if feature_cols is not None else load_feature_names()
    feat = load_stimulus_features(partition)
    images = image_list()
    feat_ids = set(feat.index.get_level_values(0))
    # official-test subjects have synthetic ids 400..447 -> route to the test pkl
    if partition == "train" and any(int(s) >= 400 for s in subject_ids):
        return subject_matrices(subject_ids, partition="test",
                                feature_cols=feature_cols)
    out = {}
    for sid in subject_ids:
        key = sid - 400 if partition == "test" else sid  # test pkl uses file ids 0..47
        sub = feat.loc[key] if key in feat_ids else None
        if sub is None or len(sub) == 0:
            X = np.full((NUM_STIMULI, len(feature_cols)), np.nan, dtype=np.float32)
            mask = np.zeros(NUM_STIMULI, dtype=bool)
        else:
            X = sub.reindex(images)[feature_cols].to_numpy(dtype=np.float32)
            mask = ~np.isnan(X).all(axis=1)
        out[int(sid)] = (X, mask)
    return out


def feature_norm_stats(subject_ids, partition="train", feature_cols=None):
    """Per-feature mean/std across the given subjects (unsupervised scaling).

    Used to scale raw inputs before deviation computation and to fill missing
    feature cells. Fits on the passed subject ids only — call with the
    training fold to avoid leakage.
    """
    mats = subject_matrices(subject_ids, partition, feature_cols=feature_cols)
    X = np.concatenate([m[0][m[1]] for m in mats.values()], axis=0)
    mean = np.nanmean(X, axis=0).astype(np.float32)
    std = np.nanstd(X, axis=0).astype(np.float32)
    std = np.maximum(std, EPS)
    return mean, std


def hc_normative_stats(train_ids, feature_cols=None):
    """Stimulus-conditioned HC normative statistics from train-fold HC only.

    Returns (mu (S,D), sigma (S,D), sigma_shrink (S,D)) with mu, sigma
    computed per stimulus over the HC subjects of `train_ids`. Stimuli with
    no HC coverage in the train fold get mu=0, sigma=EPS (recorded; such a
    stimulus is all-NaN for the affected subjects and thus masked anyway).
    """
    mats = subject_matrices(train_ids, feature_cols=feature_cols)
    hc_ids = [s for s in train_ids if s < 200]
    D = len(feature_cols) if feature_cols is not None else NUM_FEATURES
    X_all = np.stack([mats[s][0] for s in hc_ids], axis=0)          # (N_HC, S, D)
    mask_all = np.stack([mats[s][1] for s in hc_ids], axis=0)
    mu = np.full((NUM_STIMULI, D), np.nan, dtype=np.float32)
    sigma = np.full((NUM_STIMULI, D), np.nan, dtype=np.float32)
    for s in range(NUM_STIMULI):
        col = X_all[:, s, :]
        m = mask_all[:, s]
        if m.sum() == 0:
            continue
        mu[s] = np.nanmean(col[m], axis=0)
        sigma[s] = np.nanstd(col[m], axis=0)
    sigma = np.nan_to_num(sigma, nan=0.0)
    mu = np.nan_to_num(mu, nan=0.0)
    # shrinkage toward the per-feature median sigma across stimuli (Marquand-style)
    med_sigma = np.nanmedian(sigma, axis=0)
    sigma_shrink = 0.9 * sigma + 0.1 * med_sigma
    sigma_shrink = np.maximum(sigma_shrink, EPS)
    return mu, np.maximum(sigma, EPS), sigma_shrink


def fill_missing_cells(X, fill):
    """Replace NaN cells of X (S, D) with the train-fitted values `fill`.

    `fill` is either per-feature (D,) for learned mode (feature_norm_stats mu)
    or per-stimulus (S, D) for the hard-deviation modes (HC stimulus norms).
    """
    fill = np.asarray(fill, dtype=np.float32)
    if fill.ndim == 1:
        fill = fill[None, :]
    return np.where(np.isnan(X), fill, X).astype(np.float32)


def apply_deviation(X, mask, mu, sigma, sigma_shrink, mode):
    """Convert a filled subject matrix to per-stimulus deviation vectors.

    X must already have missing feature cells filled (see fill_missing_cells);
    all-NaN (missing stimulus) rows are zeroed by the mask afterwards.

    mode: 'raw'  -> standardized raw features (mu/sigma here must be the global
                     feature stats from feature_norm_stats)
          'diff' -> (x - mu) with mu = HC stimulus norms
          'z'    -> (x - mu) / sigma
          'mahal'-> z-deviation concatenated with the per-stimulus Mahalanobis
                     scalar computed with the SHRINKAGE DIAGONAL scales:
                     m_s = sqrt(mean(((x - mu) / sigma_shrink) ** 2)) over the
                     D features of the stimulus. This is a diagonal-shrinkage
                     approximation, NOT a full-covariance Mahalanobis distance.
    Returns (S, D_out, mask) with D_out = D (raw/diff/z) or D + 1 (mahal).
    """
    X = np.asarray(X, dtype=np.float32)
    if np.isnan(X).any():
        raise ValueError("apply_deviation got NaN cells; fill them first "
                         "(fill_missing_cells) — mask only marks missing stimuli")
    if mode == "raw":
        D = (X - mu) / sigma
    elif mode == "diff":
        D = X - mu
    elif mode == "z":
        D = (X - mu) / sigma
    elif mode == "mahal":
        z = (X - mu) / sigma
        m_s = np.sqrt(np.mean(((X - mu) / sigma_shrink) ** 2, axis=1, keepdims=True))
        D = np.concatenate([z, m_s.astype(np.float32)], axis=1)
    else:
        raise ValueError(mode)
    D = D * mask[:, None].astype(np.float32)
    return D, mask
