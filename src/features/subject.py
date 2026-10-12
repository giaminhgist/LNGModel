"""Subject-level feature matrices from the per-stimulus hand-crafted features.

The processed dataset is indexed by (subject_id, image) with non-contiguous
subject ids. This module builds the three subject-level representations
documented in docs/experiments/baselines.md:

  agg    : mean + std of each of the 45 features over the 100 stimuli  (91 dims
           incl. n_valid_stim)
  catagg : mean of each feature within each stimulus category          (181 dims)
  concat : raw concatenation of the 100 per-stimulus vectors           (4500 dims)

Missing (subject, stimulus) pairs are NaN and are skipped in the aggregations.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # EMS-Projects/src
from config import PROCESSED, CATEGORIES, image_category_short  # noqa: E402


def load_stimulus_features(partition="train"):
    return pd.read_pickle(PROCESSED / f"stimulus_features_{partition}.pkl")


def _subset(subject_ids, partition):
    feat = load_stimulus_features(partition)
    ids = list(subject_ids)
    if partition == "test":  # metadata test ids 400..447 map to file indices 0..47
        ids = [s - 400 for s in ids]
    feat = feat.loc[feat.index.get_level_values(0).isin(ids)]
    feat = feat.reset_index()
    feat = feat.rename(columns={"level_0": "subject_id", "level_1": "image"})
    if partition == "test":
        feat["subject_id"] = feat["subject_id"] + 400
    return feat


def _n_valid(feat):
    return feat.groupby("subject_id")["spa_fix_count"].apply(lambda s: s.notna().sum())


def build_agg(subject_ids, partition="train"):
    """mean+std over stimuli per feature -> (n_subjects, 2*45+1)."""
    feat = _subset(subject_ids, partition)
    n_valid = _n_valid(feat)
    g = feat.groupby("subject_id")
    mean = g.mean(numeric_only=True).add_prefix("mean_")
    std = g.std(numeric_only=True).add_prefix("std_")
    out = pd.concat([mean, std], axis=1).reindex(subject_ids)
    out["n_valid_stim"] = n_valid
    return out


def build_catagg(subject_ids, partition="train"):
    """mean of each feature within each stimulus category -> (n, 4*45+1)."""
    feat = _subset(subject_ids, partition)
    feat["cat"] = feat["image"].map(image_category_short)
    n_valid = _n_valid(feat)
    pieces = []
    for c in CATEGORIES:
        sub = feat[feat.cat == c].groupby("subject_id").mean(numeric_only=True)
        pieces.append(sub.add_prefix(f"{c}_"))
    out = pd.concat(pieces, axis=1).reindex(subject_ids)
    out["n_valid_stim"] = n_valid
    return out


def build_concat(subject_ids, partition="train"):
    """raw concatenation of per-stimulus vectors -> (n, 100*45), NaN->0."""
    feat = _subset(subject_ids, partition)
    wide = feat.set_index(["subject_id", "image"]).unstack("image").fillna(0.0)
    wide.columns = [f"{img}__{f}" for f, img in wide.columns]
    return wide.reindex(subject_ids)
