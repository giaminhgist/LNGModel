"""Figures 01 — original dataset: overview, coverage, distributions, scanpaths.

Reproducibility rules used here (see figure_index.md):
  - Stimulus selection: per category, the stimulus whose mean fixation count
    across all train subjects (cleaned fixations) is closest to the category
    median -> a "typical difficulty" stimulus, deterministic.
  - Subject selection (scanpath figure): among train subjects with a valid
    exposure of that stimulus, the HC subject whose fixation count is closest
    to the HC median on that stimulus, and likewise for SZ.
  - All inferential statistics (Welch t, Cohen's d) are computed on
    SUBJECT-level aggregates (n=80 per group), not on individual fixations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FIG, TAB, CAT_COLORS, HC_COLOR, SZ_COLOR, savefig,  # noqa: E402
                    panel_label, load_metadata, image_list, category_of,
                    PROCESSED, RAW)
# figures/scripts/common.py pushed src/ onto sys.path[0]; keep this directory
# first so `rawdata` resolves to figures/scripts/rawdata.py, not src/rawdata.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rawdata import load_cleaned, load_drop_log, labels_series  # noqa: E402

IMG_ROOT = RAW / "Images"
OUT = FIG
CAT_ORDER = ["social", "natural", "synthetic", "manipulated"]

# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #
def stim_of_category():
    """image -> category for all 100 stimuli."""
    return pd.Series({im: category_of(im) for im in image_list()})


def pick_stimulus_per_category(cleaned):
    """One representative stimulus per category (rule above)."""
    counts = cleaned.groupby(["IMAGE", "category"]).size().rename("n_fix")
    picks = {}
    for cat in CAT_ORDER:
        sub = counts.xs(cat, level="category").groupby("IMAGE").mean()
        med = sub.median()
        picks[cat] = (sub - med).abs().idxmin()
    return picks


def subject_stats(cleaned):
    """Subject-level stats from cleaned fixations (per-subject unit)."""
    cleaned = cleaned[cleaned.partition == "train"]
    g = cleaned.groupby("subject_id")
    s = pd.DataFrame({
        "n_fix": g.size(),
        "dur_mean": g.FIX_DURATION.mean(),
        "pup_mean": g.FIX_PUPIL.mean(),
    })
    xy = cleaned.groupby("subject_id").apply(
        lambda d: pd.Series({
            "dispersion": np.sqrt(((d[["FIX_X", "FIX_Y"]] -
                                    d[["FIX_X", "FIX_Y"]].mean()) ** 2).sum(1)).mean(),
            "n_stim": d.IMAGE.nunique(),
        }), include_groups=False)
    s = s.join(xy)
    lab = labels_series()
    s["label"] = lab.loc[s.index]
    return s


def welch_d(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    d = (a.mean() - b.mean()) / np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return p, d


def raincloud(ax, hc, sz, title, unit, letter):
    """Half-violin + box + jitter for one subject-level metric.

    Short panel title + unit (no horizontal overflow). The Welch p / Cohen's d
    statistics are reported in the caption (figure_index.md), not in the figure.
    """
    for pos, data, color in [(-0.35, hc, HC_COLOR), (0.35, sz, SZ_COLOR)]:
        parts = ax.violinplot(data, positions=[pos], vert=True, widths=0.5,
                              showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.35)
        bp = ax.boxplot(data, positions=[pos], widths=0.16, vert=True,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="k", lw=1),
                        boxprops=dict(facecolor="white", color=color),
                        whiskerprops=dict(color=color), capprops=dict(color=color))
        for elt in bp["boxes"]:
            elt.set_facecolor(color)
        rng = np.random.default_rng(0)
        ax.scatter(pos + rng.normal(0, 0.045, len(data)), data, s=7,
                   color=color, alpha=0.5, lw=0, zorder=3)
    p, d = welch_d(np.asarray(hc), np.asarray(sz))
    panel_label(ax, letter, outside=True, title=f"{title} ({unit})")
    ax.set_xticks([-0.35, 0.35])
    ax.set_xticklabels(["HC", "SZ"])


# --------------------------------------------------------------------- #
# Figure_1 dataset overview
# --------------------------------------------------------------------- #
def fig_dataset_overview():
    meta = load_metadata()
    cleaned = load_cleaned()
    cleaned = cleaned[cleaned.partition == "train"]
    cats = stim_of_category()
    picks = pick_stimulus_per_category(cleaned)

    fig = plt.figure(figsize=(13.8, 4.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.0, 0.95],
                          height_ratios=[0.75, 1.35], hspace=0.4, wspace=0.3)
    # (a) subjects per label (compact; example thumbnails below)
    ax = fig.add_subplot(gs[0, 0])
    train = meta[meta.partition == "train"]
    n_hc = int((train.label == 0).sum())
    n_sz = int((train.label == 1).sum())
    ax.barh([1, 0], [n_hc, n_sz], color=[HC_COLOR, SZ_COLOR], height=0.55)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["SZ (label 1)", "HC (label 0)"])
    ax.set_xlim(0, 90)
    for i, v in enumerate([n_sz, n_hc]):
        ax.text(v + 1, i, str(v), va="center", fontweight="bold")
    panel_label(ax, "a", outside=True, title="Subjects — EMS dataset")
    ax.set_xlabel("number of subjects")
    # example stimulus thumbnails (2x2) below (a)
    ax_ex = fig.add_subplot(gs[1, 0])
    ax_ex.axis("off")
    gsb = ax_ex.get_subplotspec().subgridspec(3, 2, height_ratios=[0.25, 1, 1],
                                              hspace=0.42, wspace=0.08)
    ax_lab = fig.add_subplot(gsb[0, :])
    ax_lab.axis("off")
    ax_lab.text(0.5, 0.5, "one example stimulus per category",
                ha="center", va="center", fontsize=8, color="#555555")
    for i, (cat, img) in enumerate(picks.items()):
        folder = {"social": "Social Scenes", "natural": "Natural Scenes",
                  "synthetic": "Synthetic Images", "manipulated": "Manipulated Images"}[cat]
        im = plt.imread(IMG_ROOT / folder / img)
        tax = fig.add_subplot(gsb[1 + i // 2, i % 2])
        tax.imshow(im)
        tax.set_xticks([])
        tax.set_yticks([])
        tax.set_title(cat.capitalize(), fontsize=8, pad=2)
        for spine in tax.spines.values():
            spine.set_color(CAT_COLORS[cat])
            spine.set_linewidth(1.6)
    # (b) stimuli per category (full height)
    axb = fig.add_subplot(gs[:, 1])
    counts = cats.value_counts().reindex(CAT_ORDER)
    colors = [CAT_COLORS[c] for c in CAT_ORDER]
    axb.bar(range(4), counts.values, color=colors, width=0.62)
    axb.set_xticks(range(4))
    axb.set_xticklabels([f"{c}\n(n={v})" for c, v in counts.items()], fontsize=9)
    axb.set_ylim(0, 36)
    panel_label(axb, "b", outside=True, title="100 stimuli, 4 categories")
    axb.set_ylabel("number of stimuli")
    for i, v in enumerate(counts.values):
        axb.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
    # (c) official fold composition
    ax = fig.add_subplot(gs[:, 2])
    fold_hc, fold_sz = {}, {}
    for fold in ["Set_0", "Set_1", "Set_2", "Set_3"]:
        sub = train[train.official_fold == fold]
        fold_hc[fold] = int((sub.label == 0).sum())
        fold_sz[fold] = int((sub.label == 1).sum())
    x = np.arange(4)
    ax.bar(x - 0.18, [fold_hc[f] for f in ["Set_0", "Set_1", "Set_2", "Set_3"]],
           width=0.36, color=HC_COLOR, label="HC")
    ax.bar(x + 0.18, [fold_sz[f] for f in ["Set_0", "Set_1", "Set_2", "Set_3"]],
           width=0.36, color=SZ_COLOR, label="SZ")
    ax.set_xticks(x)
    ax.set_xticklabels(["Set_0", "Set_1", "Set_2", "Set_3"])
    ax.set_ylim(0, 26)
    panel_label(ax, "c", outside=True, title="Official 4-fold split")
    ax.set_ylabel("subjects per fold")
    ax.legend(loc="upper right")
    savefig(fig, OUT, "Figure_1")
    # tables
    pd.DataFrame({f: [fold_hc[f], fold_sz[f]] for f in ["Set_0", "Set_1", "Set_2", "Set_3"]},
                 index=["HC", "SZ"]).to_csv(TAB / "T01.01_fold_composition.csv")
    pd.DataFrame({"category": CAT_ORDER,
                  "n_stimuli": counts.values.tolist(),
                  "example_stimulus": [picks[c] for c in CAT_ORDER]}) \
        .to_csv(TAB / "T01.01_stimulus_counts.csv", index=False)


# --------------------------------------------------------------------- #
# Figure_2 gaze signatures: subject-level distributions + scanpaths (merged)
# --------------------------------------------------------------------- #
def fig_gaze_signatures():
    """Figure_2: top row = subject-level HC/SZ distributions,
    bottom two rows = scanpaths of HC/SZ on the same representative stimuli.
    Welch p / Cohen's d are reported in the caption, not inside the figure.
    """
    cleaned = load_cleaned()
    cleaned = cleaned[cleaned.partition == "train"]
    picks = pick_stimulus_per_category(cleaned)
    s = subject_stats(cleaned)
    hc = s[s.label == 0]
    sz = s[s.label == 1]
    fig = plt.figure(figsize=(13.8, 9.8))
    # top: distributions (a)-(c) — dispersion stats are in the caption
    gs_top = fig.add_gridspec(1, 3, top=0.94, bottom=0.69, left=0.08,
                              right=0.96, wspace=0.25)
    axes = [fig.add_subplot(gs_top[0, i]) for i in range(3)]
    # bottom: scanpath grid
    gs = fig.add_gridspec(2, 4, top=0.62, bottom=0.03, left=0.08,
                          right=0.96, hspace=0.035, wspace=0.08)
    raincloud(axes[0], hc.n_fix, sz.n_fix, "Fixations per subject",
              "count", "a")
    raincloud(axes[1], hc.dur_mean, sz.dur_mean, "Mean fixation duration",
              "ms", "b")
    raincloud(axes[2], hc.pup_mean, sz.pup_mean, "Mean pupil size",
              "a.u.", "c")
    # bottom rows: scanpaths of HC (top) and SZ (bottom) on the same stimuli
    for ci, cat in enumerate(CAT_ORDER):
        stim = picks[cat]
        folder = {"social": "Social Scenes", "natural": "Natural Scenes",
                  "synthetic": "Synthetic Images", "manipulated": "Manipulated Images"}[cat]
        img = plt.imread(IMG_ROOT / folder / stim)
        sub = cleaned[(cleaned.IMAGE == stim)]
        counts = sub.groupby("subject_id").size()
        for gi, (label, color) in enumerate([("HC", HC_COLOR), ("SZ", SZ_COLOR)]):
            lab = 0 if label == "HC" else 1
            lab_s = labels_series()
            grp_counts = counts[lab_s.loc[counts.index].values == lab]
            sid = (grp_counts - grp_counts.median()).abs().idxmin()
            d = sub[sub.subject_id == sid].sort_values("FIX_INDEX")
            ax = fig.add_subplot(gs[gi, ci])
            ax.imshow(img)
            ax.plot(d.FIX_X, d.FIX_Y, "-", color=color, lw=1.1, alpha=0.9, zorder=2)
            ax.scatter(d.FIX_X.iloc[0], d.FIX_Y.iloc[0], s=60, marker="s",
                       facecolor="none", edgecolor=color, lw=1.6, zorder=3)
            ax.scatter(d.FIX_X.iloc[1:], d.FIX_Y.iloc[1:], s=10, color=color,
                       alpha=0.85, zorder=3)
            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(img.shape[0], 0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{label} subj {sid} — n={len(d)} fixations",
                         fontsize=8.5)
            if ci == 0:
                ax.set_ylabel(label, fontsize=10, color=color, fontweight="bold")
            if gi == 1:
                ax.set_xlabel(f"{stim} ({cat})", fontsize=9)
    # single block label (d) for the scanpath section (no per-cell letters)
    fig.text(0.08, 0.628, "d", fontweight="bold", fontsize=11,
             ha="left", va="bottom")
    fig.text(0.091, 0.628, "Scanpaths — HC (top) / SZ (bottom), same stimuli",
             fontsize=10, color="#333333", ha="left", va="bottom")
    savefig(fig, OUT, "Figure_2")
    s[["n_fix", "dur_mean", "pup_mean", "dispersion", "n_stim", "label"]] \
        .to_csv(TAB / "T01.03_subject_stats.csv")
    pd.DataFrame({"category": CAT_ORDER, "stimulus": [picks[c] for c in CAT_ORDER]}) \
        .to_csv(TAB / "T01.04_selected_stimuli.csv", index=False)


if __name__ == "__main__":
    fig_dataset_overview()
    fig_gaze_signatures()
    print("done 01_dataset")
