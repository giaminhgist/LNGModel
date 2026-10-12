"""EMS preprocessing + hand-crafted feature computation.

Pipeline:
  1. load every subject's fixation xlsx (Train_Valid + official Test)
  2. clean: drop off-screen / implausible-duration / pupil-outlier fixations
  3. compute the 45 hand-crafted features per (subject, stimulus)
  4. store with a MultiIndex (subject_id, image) preserving non-contiguous ids
  5. quality check: NaN audit, coordinate-range audit, report

Outputs (processed_dataset/):
  stimulus_features_train.pkl / stimulus_features_test.pkl
  metadata.csv            (subject_id, partition, label, official fold membership)
  feature_names.txt
  quality_report.txt

Usage:
    python src/scripts/preprocess.py
"""
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED, RAW, SCREEN_W, SCREEN_H  # noqa: E402
from rawdata import (load_all, image_categories, official_folds_raw,  # noqa: E402
                     subject_label)
from features.stimulus import compute_stimulus_features  # noqa: E402

DUR_MIN, DUR_MAX = 40.0, 2000.0      # plausible fixation duration window (ms)
PUPIL_SD = 4.0                        # drop fixations beyond ±4 SD of stimulus pupil
MIN_FIX = 2                           # minimum fixations for a valid (subject, stimulus)


def clean_fixations(df: pd.DataFrame):
    """Apply the cleaning rules (README.md, Dataset section) to one subject's fixations."""
    before = len(df)
    df = df[(df.FIX_X >= 0) & (df.FIX_X < SCREEN_W)
            & (df.FIX_Y >= 0) & (df.FIX_Y < SCREEN_H)].copy()
    after_screen = len(df)
    df = df[(df.FIX_DURATION > DUR_MIN) & (df.FIX_DURATION <= DUR_MAX)]
    after_dur = len(df)
    # per-stimulus pupil outliers
    med = df.groupby("IMAGE")["FIX_PUPIL"].transform("median")
    sd = df.groupby("IMAGE")["FIX_PUPIL"].transform("std").fillna(0.0)
    df = df[np.abs(df.FIX_PUPIL - med) <= PUPIL_SD * np.maximum(sd, 1e-6)]
    return df, {"raw": before, "off_screen_dropped": before - after_screen,
                "duration_dropped": after_screen - after_dur,
                "pupil_dropped": after_dur - len(df)}


def compute_features_for_subject(subj_df: pd.DataFrame, images):
    """Compute the 45 features for every stimulus of one subject."""
    rows = {}
    for image, grp in subj_df.groupby("IMAGE"):
        if len(grp) < MIN_FIX:
            continue
        grp = grp.sort_values("FIX_INDEX")
        rows[image] = compute_stimulus_features(
            grp.FIX_X.values, grp.FIX_Y.values,
            grp.FIX_DURATION.values.astype(np.float64),
            grp.FIX_PUPIL.values.astype(np.float64))
    if not rows:
        return pd.DataFrame()
    feat = pd.DataFrame(rows).T
    feat.index.name = "image"
    return feat


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    PROCESSED.mkdir(parents=True, exist_ok=True)  # create before writing
    images = image_categories().index.tolist()
    log_lines = [f"EMS preprocessing log", "=" * 40]

    for partition in ["train", "test"]:
        print(f"=== partition: {partition} ===")
        subjects = load_all(partition).groupby("subject_id")
        frames, drop_log = [], []
        for i, (sid, subj) in enumerate(subjects, 1):
            if i % 25 == 0 or i == len(subjects):
                print(f"  {partition}: {i}/{len(subjects)} subjects", flush=True)
            clean, d = clean_fixations(subj)
            d["subject_id"] = sid
            drop_log.append(d)
            feat = compute_features_for_subject(clean, images)
            if len(feat):
                feat["subject_id"] = sid
                feat["image"] = feat.index
                frames.append(feat.reset_index(drop=True))
        table = pd.concat(frames, ignore_index=True)
        table = table.set_index(["subject_id", "image"]).sort_index()
        table = table.reindex(
            pd.MultiIndex.from_product(
                [sorted(table.index.get_level_values(0).unique()), images],
                names=["subject_id", "image"]))
        out = PROCESSED / f"stimulus_features_{partition}.pkl"
        table.to_pickle(out)

        drops = pd.DataFrame(drop_log).set_index("subject_id")
        log_lines += [
            f"\n[{partition}] {len(table.index.get_level_values(0).unique())} subjects "
            f"x {len(images)} stimuli -> {table.shape[0]} rows, {table.shape[1]} cols",
            f"cleaning dropped per subject (mean): "
            f"off_screen={drops.off_screen_dropped.mean():.1f}, "
            f"duration={drops.duration_dropped.mean():.1f}, "
            f"pupil={drops.pupil_dropped.mean():.1f}",
            f"total dropped: {drops[['off_screen_dropped','duration_dropped','pupil_dropped']].sum().to_dict()}",
        ]
        print(f"saved {out}  shape={table.shape}")

    # ---- metadata ----
    # NOTE: official test files are named Test_000..Test_047, so their *file
    # indices* (0..47) collide with train subject ids (0..199). Test rows get
    # synthetic unique subject ids 400..447; the original file index is kept
    # in `file_id` (this is the id used in the official prob_test.txt).
    folds = official_folds_raw()
    rows = []
    for sid in sorted(set(folds["Set_0"]) | set(folds["Set_1"]) | set(folds["Set_2"]) | set(folds["Set_3"])):
        fold = next(k for k, v in folds.items() if sid in v)
        rows.append({"subject_id": sid, "partition": "train", "label": subject_label(sid),
                     "official_fold": fold, "file_id": np.nan})
    test_files = sorted((RAW / "Test" / "Fixations").glob("*.xlsx"))
    for i, f in enumerate(test_files):
        rows.append({"subject_id": 400 + i, "partition": "test", "label": np.nan,
                     "official_fold": np.nan,
                     "file_id": int(f.stem.split("_")[-1])})
    meta = pd.DataFrame(rows).set_index("subject_id").sort_index()
    meta.to_csv(PROCESSED / "metadata.csv")
    log_lines.append(f"\nmetadata: {meta.shape[0]} subjects (train={meta.partition.eq('train').sum()}, "
                     f"test={meta.partition.eq('test').sum()})")

    # ---- feature names ----
    feat = pd.read_pickle(PROCESSED / "stimulus_features_train.pkl")
    (PROCESSED / "feature_names.txt").write_text(
        "\n".join(feat.columns.tolist()) + "\n")
    log_lines.append(f"features ({len(feat.columns)}): {feat.columns.tolist()}")

    # ---- quality checks ----
    log_lines += ["\n=== quality checks ==="]
    for partition in ["train", "test"]:
        t = pd.read_pickle(PROCESSED / f"stimulus_features_{partition}.pkl")
        n_total = t.shape[0] * t.shape[1]
        n_nan = int(t.isna().sum().sum())
        log_lines.append(
            f"[{partition}] NaN cells: {n_nan}/{n_total} ({n_nan/n_total*100:.3f}%)  "
            f"| empty (subject,stimulus) pairs: {int(t.isna().all(axis=1).sum())}")
        top_nan = t.isna().mean().sort_values(ascending=False).head(5)
        log_lines.append(f"[{partition}] features with most NaN: "
                         + ", ".join(f"{k}={v*100:.1f}%" for k, v in top_nan.items()))
        # coordinate sanity: mean_x must lie inside the screen
        for c in ["spa_mean_x", "spa_mean_y"]:
            v = t[c].dropna()
            if len(v):
                log_lines.append(f"[{partition}] {c}: min={v.min():.1f} max={v.max():.1f} "
                                 f"(screen {SCREEN_W}x{SCREEN_H})")
        # off-screen leftover check
        bad = ((t[["spa_mean_x", "spa_mean_y"]].dropna().spa_mean_x < 0)
               | (t[["spa_mean_x", "spa_mean_y"]].dropna().spa_mean_x >= SCREEN_W)).sum()
        log_lines.append(f"[{partition}] centroid outside screen: {int(bad)}")

    report = "\n".join(log_lines)
    (PROCESSED / "quality_report.txt").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
