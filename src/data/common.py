"""Shared data utilities for the normative-gaze research codebase.

Paths, subject/fold utilities, and the official 4-fold evaluation protocol.
Reuses the processed dataset (stimulus features + metadata) produced by
src/scripts/preprocess.py. Shared constants live in src/config.py (re-exported
here so existing import sites keep working).
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # EMS-Projects/src
from config import (ROOT, PROCESSED, RAW, OUTPUTS,  # noqa: E402
                    NUM_STIMULI, NUM_FEATURES, SCREEN_W, SCREEN_H)


def load_metadata():
    meta = pd.read_csv(PROCESSED / "metadata.csv", index_col="subject_id")
    return meta


def train_subject_ids():
    meta = load_metadata()
    return meta[meta.partition == "train"].index.tolist()


def official_folds():
    """Yield (fold_name, train_ids, val_ids) for the official 4-fold protocol."""
    meta = load_metadata()
    meta = meta[meta.partition == "train"]
    folds = {}
    for name in ["Set_0", "Set_1", "Set_2", "Set_3"]:
        folds[name] = meta[meta.official_fold == name].index.tolist()
    for name, val_ids in folds.items():
        train_ids = [s for n, ids in folds.items() if n != name for s in ids]
        yield name, train_ids, val_ids


def labels_of(subject_ids):
    meta = load_metadata()
    return meta.loc[subject_ids, "label"].to_numpy(dtype=np.int64)


def load_stimulus_features(partition="train"):
    """DataFrame (subject_id, image) x 45 features; NaN for missing pairs."""
    return pd.read_pickle(PROCESSED / f"stimulus_features_{partition}.pkl")


def load_feature_names():
    """Ordered list of the 45 base feature columns of the processed schema."""
    names = [ln.strip() for ln in (PROCESSED / "feature_names.txt").read_text().splitlines()
             if ln.strip()]
    return names


def data_identity():
    """Content hash of the processed data artifacts + schema.

    Recorded in every run so a resume/skip decision can detect data changes.
    """
    h = hashlib.sha256()
    for fname in ["stimulus_features_train.pkl", "stimulus_features_test.pkl",
                  "metadata.csv", "feature_names.txt"]:
        p = PROCESSED / fname
        if not p.exists():
            return {"data_hash": "missing", "schema_hash": "missing"}
        h.update(fname.encode())
        h.update(p.read_bytes())
    schema_hash = hashlib.sha256(
        "\n".join(load_feature_names()).encode()).hexdigest()
    return {"data_hash": h.hexdigest(), "schema_hash": schema_hash}


def image_list():
    feat = load_stimulus_features("train")
    return sorted(feat.index.get_level_values(1).unique())
