"""Shared constants: repo paths, screen geometry, stimulus-category maps.

Single source of truth merged (restructure 2026-10) from the former
src/common.py, src/data/common.py and src/baseline/common.py.

Two category mappings are kept verbatim from their original modules because
their key formats differ and both are in active use:
  - CATEGORY_PREFIXES / image_category()      — underscored prefixes ("act_"),
    raw-data side (former src/common.py).
  - PREFIX_CATEGORY / image_category_short()  — token prefixes ("act"),
    subject-level feature side (former src/baseline/common.py).
Do not merge the two lookups; behavior must stay bit-for-bit identical.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root, portable across machines
RAW = ROOT / "original_dataset" / "EMS"
IMAGES_DIR = RAW / "Images"
PROCESSED = ROOT / "processed_dataset"
OUTPUTS = ROOT / "outputs"
RESULTS = OUTPUTS / "experiments"           # run outputs (former baseline RESULTS)

NUM_STIMULI = 100
NUM_FEATURES = 45
CATEGORIES = ["social", "natural", "synthetic", "manipulated"]

SCREEN_W, SCREEN_H = 1024, 768  # Eyelink display resolution used in EMS

# stimulus name -> category (derived from the official Images/ folder layout);
# underscored-prefix form used by raw-data loading (former src/common.py)
CATEGORY_PREFIXES = {
    "act_": "social", "por_": "social", "soc_": "social",
    "ind_": "natural", "land_": "natural", "outman_": "natural", "sat_": "natural",
    "art_": "synthetic", "cat_": "synthetic", "pat_": "synthetic",
    "low_": "manipulated", "mood_": "manipulated", "noi_": "manipulated",
    "patch_": "manipulated", "rand_": "manipulated",
}

# token-prefix form (no underscore) used by subject-level feature builders
# (former src/baseline/common.py) — same mapping, different key format
PREFIX_CATEGORY = {
    "act": "social", "por": "social", "soc": "social",
    "ind": "natural", "land": "natural", "outman": "natural", "sat": "natural",
    "art": "synthetic", "cat": "synthetic", "pat": "synthetic",
    "low": "manipulated", "mood": "manipulated", "noi": "manipulated",
    "patch": "manipulated", "rand": "manipulated",
}


def image_category(name: str) -> str:
    """Map an image file name (with or without .jpg) to its stimulus category."""
    stem = name.replace(".jpg", "").replace(".jpeg", "")
    for prefix, cat in CATEGORY_PREFIXES.items():
        if stem.startswith(prefix):
            return cat
    raise KeyError(f"unknown stimulus prefix in {name}")


def image_category_short(name):
    """Token-prefix category lookup (former src/baseline/common.py::image_category)."""
    return PREFIX_CATEGORY[name.replace(".jpg", "").split("_")[0]]
