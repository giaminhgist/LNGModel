"""Cached access to raw + cleaned fixations for the dataset figures.

Loads every subject's fixation xlsx once, applies the SAME cleaning rules as
src/scripts/preprocess.py::clean_fixations, and caches the cleaned long-format
table as pickle under figures/cache/.

The cleaning function is inlined (identical rules to
src/scripts/preprocess.py: DUR_MIN 40, DUR_MAX 2000, pupil ±4 SD within
stimulus, screen bounds 1024x768) to avoid the module-name collision between
figures/scripts/rawdata.py and src/rawdata.py inside the preprocess import.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CACHE, src_common  # noqa: E402

CLEAN_PQ = CACHE / "fixations_cleaned.pkl"
DROP_PQ = CACHE / "fixations_drop_log.pkl"


def _clean_fixations(df: pd.DataFrame):
    """Mirror of src/scripts/preprocess.py::clean_fixations (see module docstring)."""
    SCREEN_W, SCREEN_H = 1024.0, 768.0
    before = len(df)
    df = df[(df.FIX_X >= 0) & (df.FIX_X < SCREEN_W)
            & (df.FIX_Y >= 0) & (df.FIX_Y < SCREEN_H)].copy()
    after_screen = len(df)
    df = df[(df.FIX_DURATION > 40.0) & (df.FIX_DURATION <= 2000.0)]
    after_dur = len(df)
    med = df.groupby("IMAGE")["FIX_PUPIL"].transform("median")
    sd = df.groupby("IMAGE")["FIX_PUPIL"].transform("std").fillna(0.0)
    df = df[np.abs(df.FIX_PUPIL - med) <= 4.0 * np.maximum(sd, 1e-6)]
    return df, {"raw": before, "off_screen_dropped": before - after_screen,
                "duration_dropped": after_screen - after_dur,
                "pupil_dropped": after_dur - len(df)}


def build_cache(partitions=("train", "test"), force=False):
    """Load + clean all subjects of the given partitions; write cache files."""
    if CLEAN_PQ.exists() and DROP_PQ.exists() and not force:
        return
    frames, drops = [], []
    for partition in partitions:
        raw = src_common.load_all(partition)
        raw["partition"] = partition
        raw = raw.sort_values(["subject_id", "IMAGE", "FIX_INDEX"]).reset_index(drop=True)
        for sid, subj in raw.groupby("subject_id"):
            clean, d = _clean_fixations(subj)
            d["subject_id"] = sid
            drops.append(d)
            if len(clean):
                frames.append(clean)
    cleaned = pd.concat(frames, ignore_index=True)
    cleaned = cleaned.sort_values(["subject_id", "IMAGE", "FIX_INDEX"]).reset_index(drop=True)
    cleaned.to_pickle(CLEAN_PQ)
    pd.DataFrame(drops).set_index("subject_id").to_pickle(DROP_PQ)
    print(f"cache written: {len(cleaned)} fixations, "
          f"{cleaned.subject_id.nunique()} subjects -> {CLEAN_PQ}")


def load_cleaned():
    if not CLEAN_PQ.exists():
        build_cache()
    return pd.read_pickle(CLEAN_PQ)


def load_drop_log():
    if not DROP_PQ.exists():
        build_cache()
    return pd.read_pickle(DROP_PQ)


def labels_series():
    """subject_id -> label (0 HC / 1 SZ) for the train partition."""
    return src_common.subject_labels()


if __name__ == "__main__":
    build_cache()
