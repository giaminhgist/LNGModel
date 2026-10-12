"""Raw EMS data loading: fixation tables, subject ids/labels, raw folds.

Loads the original EMS release (original_dataset/EMS/) — one fixation xlsx
per subject, stimulus categories, and the official Train_Valid.xlsx fold
assignment. Shared constants live in src/config.py.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # EMS-Projects/src
from config import RAW, IMAGES_DIR, image_category  # noqa: E402

TRAIN_FIX_DIR = RAW / "Train_Valid" / "Fixations"
TEST_FIX_DIR = RAW / "Test" / "Fixations"


def image_categories() -> pd.Series:
    """Series mapping every stimulus image name -> category (100 images)."""
    names = sorted(p.name for p in IMAGES_DIR.glob("*/*.jpg"))
    return pd.Series({n: image_category(n) for n in names}).sort_index()


def subject_label(subject_id: int) -> int:
    """EMS convention: ids < 200 are HC (label 0), ids >= 200 are SZ (label 1)."""
    return 0 if int(subject_id) < 200 else 1


def load_subject(subject_id, partition="train"):
    """Load one subject's fixation xlsx into a DataFrame with extra columns."""
    fix_dir = TRAIN_FIX_DIR if partition == "train" else TEST_FIX_DIR
    fname = f"{int(subject_id):03d}.xlsx" if partition == "train" else f"Test_{subject_id}.xlsx"
    df = pd.read_excel(fix_dir / fname)
    df["subject_id"] = subject_id
    df["category"] = df["IMAGE"].map(image_category)
    return df


def load_all(partition="train", subjects=None):
    """Load all subject files of one partition into one long DataFrame.

    partition: 'train' -> 160 Train_Valid subjects, 'test' -> 48 official test subjects.
    subjects: optional list of subject ids to restrict loading.
    """
    fix_dir = TRAIN_FIX_DIR if partition == "train" else TEST_FIX_DIR
    files = sorted(fix_dir.glob("*.xlsx"))
    if subjects is not None:
        files = [f for f in files if int(f.stem.split("_")[-1]) in set(subjects)]
    frames = []
    for f in files:
        sid = int(f.stem.split("_")[-1])
        df = pd.read_excel(f)
        df["subject_id"] = sid
        df["category"] = df["IMAGE"].map(image_category)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def official_folds_raw() -> pd.DataFrame:
    """Official 4-fold assignment (Train_Valid.xlsx): Set_0 .. Set_3, 40 subjects each.

    Raw-xlsx version used by preprocessing to build processed_dataset/metadata.csv;
    experiment code uses the metadata-based data.common.official_folds().
    """
    df = pd.read_excel(RAW / "Train_Valid.xlsx")
    folds = {}
    for col in df.columns:
        folds[col] = [int(x) for x in df[col].dropna().tolist()]
    return folds


def train_subject_ids():
    return sorted(int(f.stem) for f in TRAIN_FIX_DIR.glob("*.xlsx"))


def test_subject_ids():
    return sorted(int(f.stem.split("_")[-1]) for f in TEST_FIX_DIR.glob("*.xlsx"))


def subject_labels(subject_ids=None):
    """Series subject_id -> label for train/valid subjects (ids < 200 = HC)."""
    ids = train_subject_ids() if subject_ids is None else subject_ids
    return pd.Series({s: subject_label(s) for s in ids}, name="label").sort_index()
