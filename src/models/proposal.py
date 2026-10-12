"""Main proposal — Learned Stimulus-Conditioned Normative Modeling.

Encoder f_theta maps each stimulus feature vector to a latent z; a HC latent
normative bank (per-stimulus mu^z, sigma^z) is recomputed EVERY EPOCH from the
train-fold HC encodings (model.refresh_bank hook, called by the trainer);
a comparator produces per-stimulus deviation d_{i,s}; a set-level pooling
aggregates the S deviations into the subject embedding; an MLP head classifies.

Hard-deviation ablations (deviation='z' | 'diff' | 'mahal') bypass the
encoder/comparator: per-stimulus deviations from HC normative statistics are
computed directly in raw feature space by the dataset, then fed through the
SAME pooling + head as the learned path.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # EMS-Projects/src
from data.common import labels_of  # noqa: E402
from data.tabular import (subject_matrices, feature_norm_stats,  # noqa: E402
                          hc_normative_stats, apply_deviation,
                          fill_missing_cells)

EPS = 1e-6


class NormativeDataset:
    """Subjects with per-stimulus deviation tensors + masks.

    deviation='learned': input features standardized per-feature by stats
    fitted on the train subjects only (unsupervised, leakage-safe); the
    encoder + latent bank do the deviation work inside the model.
    deviation='z'|'diff'|'mahal': hard per-stimulus deviations from HC
    normative statistics (fit on train-fold HC only), D_out = D (z/diff)
    or D + 1 (mahal, z concatenated with the shrinkage Mahalanobis scalar).

    Missingness: missing FEATURE cells are filled with the train-fitted
    per-feature mean (learned) / HC stimulus norm (hard modes) so they
    contribute zero deviation; missing STIMULI (all-NaN rows) stay masked.
    See src/data/tabular.py for the full rule.
    """

    def __init__(self, subject_ids, train_ids_for_stats, deviation="learned",
                 stim_subset=None, feature_cols=None):
        self.subjects = list(subject_ids)
        self.labels = labels_of(self.subjects)
        self.deviation = deviation
        self.stim_subset = stim_subset  # optional list of stimulus indices
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        mats = subject_matrices(self.subjects, feature_cols=self.feature_cols)
        if deviation == "learned":
            mu, sigma = feature_norm_stats(train_ids_for_stats,
                                           feature_cols=self.feature_cols)
            sigma_shrink = sigma
            fill = mu  # per-feature (D,) means from train subjects
        else:
            mu, sigma, sigma_shrink = hc_normative_stats(train_ids_for_stats,
                                                         feature_cols=self.feature_cols)
            fill = mu  # per-stimulus (S, D) HC norms
        self.X, self.mask = [], []
        for s in self.subjects:
            X, mask = mats[s]
            if self.stim_subset is not None:
                X = X[self.stim_subset]
                mask = mask[self.stim_subset]
                if deviation == "learned":  # feature stats are per-feature (D,)
                    mu_s, sigma_s, sigshrink_s = mu, sigma, sigma_shrink
                    fill_s = fill
                else:                       # HC norms are per-stimulus (S, D)
                    mu_s = mu[self.stim_subset]
                    sigma_s = sigma[self.stim_subset]
                    sigshrink_s = sigma_shrink[self.stim_subset]
                    fill_s = fill[self.stim_subset]
            else:
                mu_s, sigma_s, sigshrink_s = mu, sigma, sigma_shrink
                fill_s = fill
            X = fill_missing_cells(X, fill_s)
            if deviation == "learned":
                D = (X - mu_s) / sigma_s
                D = D * mask[:, None].astype(np.float32)
            else:
                D, mask = apply_deviation(X, mask, mu_s, sigma_s, sigshrink_s,
                                          deviation)
            self.X.append(torch.from_numpy(D.astype(np.float32)))
            self.mask.append(torch.from_numpy(mask.astype(np.float32)))

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        return (self.X[idx], self.mask[idx]), self.labels[idx], self.subjects[idx]

    def collate(self, batch):
        return (torch.stack([b[0] for b in batch], 0),
                torch.stack([b[1] for b in batch], 0))

    def hc_indices(self):
        return [i for i, s in enumerate(self.subjects) if s < 200]


class NormativeModel(nn.Module):
    """Encoder + HC latent bank + comparator + set pooling + head.

    deviation:   'learned' (default, full pipeline) | 'z' | 'diff' | 'mahal'
                 (hard deviation in feature space, no encoder/comparator/bank)
    comparator:  'mlp' (g_phi([z, mu, z-mu, z*mu])) | 'sub' (z-mu) | 'zsub'
    pool:        'attention' | 'mean' | 'deepset' (mean||max) | 'max'
                 (masked max over stimuli)
    """

    def __init__(self, d_in=45, latent_dim=128, dev_dim=64, comparator="mlp",
                 pool="attention", dropout=0.3, deviation="learned",
                 n_stim=100):
        super().__init__()
        self.deviation = deviation
        self.pool = pool
        if deviation == "learned":
            self.latent_dim = latent_dim
            self.comparator = comparator
            self.dev_dim = dev_dim
            self.encoder = nn.Sequential(
                nn.Linear(d_in, 128), nn.LayerNorm(128), nn.GELU(),
                nn.Linear(128, latent_dim))
            if comparator == "mlp":
                self.comp = nn.Sequential(
                    nn.Linear(latent_dim * 4, 256), nn.ReLU(),
                    nn.Linear(256, 128), nn.ReLU(),
                    nn.Linear(128, dev_dim))
            else:  # 'sub' | 'zsub'
                self.proj = nn.Linear(latent_dim, dev_dim)
            # HC latent normative bank (per stimulus) — refreshed every epoch
            self.register_buffer("bank_mu", torch.zeros(n_stim, latent_dim))
            self.register_buffer("bank_sigma", torch.ones(n_stim, latent_dim))
        else:  # hard deviations arrive pre-computed from the dataset
            self.dev_dim = d_in + (1 if deviation == "mahal" else 0)
        if pool == "attention":
            self.scorer = nn.Sequential(nn.Linear(self.dev_dim, 32), nn.ReLU(),
                                        nn.Linear(32, 1))
        self.head = nn.Sequential(
            nn.Linear(self.dev_dim * (2 if pool == "deepset" else 1), 64),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    # ------------------------------------------------------------------ #
    # bank refresh hook (trainer calls it once per epoch, before training;
    # hard-deviation modes have no bank -> no-op).
    # Schedule (locked, lagged by one epoch): the bank used during epoch e's
    # training AND for the epoch-e train/val evaluation is computed at the
    # START of epoch e from the pre-epoch-e weights. The best checkpoint's
    # state_dict therefore contains exactly the bank its validation metrics
    # were computed with, and restoring it reproduces those metrics.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def refresh_bank(self, train_ds):
        if self.deviation != "learned":
            return
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        hc = train_ds.hc_indices()
        D = torch.stack([train_ds[i][0][0] for i in hc], 0).to(device)
        mask = torch.stack([train_ds[i][0][1] for i in hc], 0).to(device)
        z, _ = self._encode(D, mask)
        valid = mask[..., None]
        mu = (z * valid).sum(0) / valid.sum(0).clamp(min=1)          # (S, L)
        var = ((z - mu[None]) ** 2 * valid).sum(0) / valid.sum(0).clamp(min=1)
        self.bank_mu.copy_(mu)
        self.bank_sigma.copy_(torch.sqrt(var.clamp(min=EPS)) + EPS)
        self.train(was_training)

    # ------------------------------------------------------------------ #
    def _encode(self, D, mask):
        B, S, _ = D.shape
        z = self.encoder(D.view(B * S, -1)).view(B, S, self.latent_dim)
        return z, mask

    def _deviate(self, z, mask):
        B, S, L = z.shape
        mu = self.bank_mu[None].expand(B, S, L)
        sig = self.bank_sigma[None].expand(B, S, L)
        if self.comparator == "mlp":
            feat = torch.cat([z, mu, z - mu, z * mu], dim=-1)
            d = self.comp(feat)
        elif self.comparator == "sub":
            d = self.proj(z - mu)
        else:  # zsub
            d = self.proj((z - mu) / (sig + EPS))
        d = d * mask[..., None]
        return d

    def _pool(self, d, mask):
        valid = mask[..., None]
        if self.pool == "mean":
            return (d * valid).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        if self.pool == "max":
            return (d * valid + (1 - valid) * (-1e9)).max(1).values
        if self.pool == "deepset":
            mean = (d * valid).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            maxv = (d * valid + (1 - valid) * (-1e9)).max(1).values
            return torch.cat([mean, maxv], dim=1)
        score = self.scorer(d) + (mask - 1)[..., None] * 1e9
        a = torch.softmax(score, dim=1)
        return (a * d * valid).sum(1)

    def forward(self, x, y=None):
        D, mask = x
        if self.deviation == "learned":
            z, mask = self._encode(D, mask)
            d = self._deviate(z, mask)
        else:
            d = D
        h = self._pool(d, mask)
        prob = torch.sigmoid(self.head(h))
        return prob, {"emb": h}
