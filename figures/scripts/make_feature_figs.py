"""Figure 3 — HC vs SZ effect sizes for the 45 hand-crafted features.

Statistical unit: SUBJECT-level aggregation of each feature over valid
stimuli (mean), n=80 per group — never individual fixations or
subject-stimulus pairs treated as independent.

Implementation notes honored here (see src/features/stimulus.py):
  - tem_ifi_* = mean of (dur_i + dur_{i+1})/2 (onset timestamps unavailable)
  - tem_velocity_mean = saccade amplitude / that IFI proxy (px/ms) — NOT a
    measured saccade velocity
  - tem_fix_rate = n_fix / (sum(dur) + mean_ifi*(n-1)) * 1000  (fix/s, time
    reconstructed in code)
  - tem_trans_entropy = occupancy entropy of fixations minus the last one on a
    4x3 grid — NOT a transition-matrix entropy
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FIG, TAB, HC_COLOR, SZ_COLOR, FEATURE_NAMES,  # noqa: E402
                    FEATURE_GROUPS, feature_group_of, savefig,
                    panel_label, load_subject_features, load_metadata,
                    category_of)

OUT = FIG
CAT_ORDER = ["social", "natural", "synthetic", "manipulated"]

meta = load_metadata()
LABELS = meta.loc[meta.partition == "train", "label"]


def subject_level():
    """Subject x 45 feature table (mean over valid stimuli)."""
    feat = load_subject_features("train")
    return feat.groupby(level="subject_id").mean()


def welch_d(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    d = (a.mean() - b.mean()) / np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return p, d


# --------------------------------------------------------------------- #
# Figure 3 — HC vs SZ effect sizes + rainclouds + category profiles
# --------------------------------------------------------------------- #
def fig_effect_sizes():
    S = subject_level().reindex(columns=FEATURE_NAMES)
    hc = S[LABELS == 0]
    sz = S[LABELS == 1]
    rows = []
    for f in FEATURE_NAMES:
        a, b = hc[f].dropna(), sz[f].dropna()
        p, d = welch_d(a, b)
        rows.append({"feature": f, "group": feature_group_of(f),
                     "cohen_d": d, "p_welch": p,
                     "hc_mean": a.mean(), "sz_mean": b.mean(),
                     "hc_n": len(a), "sz_n": len(b)})
    es = pd.DataFrame(rows).set_index("feature")
    es = es.sort_values("cohen_d")
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(13.8, 9.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.26,
                          left=0.06, right=0.975, top=0.955, bottom=0.04)
    gs_top = gs[0].subgridspec(1, 2, width_ratios=[1.25, 1], wspace=0.28)
    ax = fig.add_subplot(gs_top[0])
    colors = [plt.cm.tab10(list(FEATURE_GROUPS).index(feature_group_of(f)))
              for f in es.index]
    ax.barh(range(45), es.cohen_d.values, color=colors)
    for i, (f, row) in enumerate(es.iterrows()):
        stars = "***" if row.p_welch < 0.001 else ("**" if row.p_welch < 0.01 else
                                                    ("*" if row.p_welch < 0.05 else "n.s."))
        x = row.cohen_d + (0.02 if row.cohen_d >= 0 else -0.02)
        ha = "left" if row.cohen_d >= 0 else "right"
        ax.text(x, i, stars, va="center", ha=ha, fontsize=6.5,
                color="#333333" if stars != "n.s." else "#aaaaaa")
    ax.set_yticks(range(45))
    ax.set_yticklabels(es.index, fontsize=6.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Cohen's d (HC − SZ, subject level; positive = HC larger)")
    panel_label(ax, "a", outside=True, title="Effect size of every feature")
    # rainclouds for representative features (one per group)
    ax = fig.add_subplot(gs_top[1])
    reps = ["spa_dispersion", "spa_entropy", "geo_sacc_amp_mean", "tem_dur_mean", "pup_mean"]
    # first grid row is an empty spacer so the panel label above the
    # container does not touch the top raincloud
    gs_rain = gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=ax.get_subplotspec(),
                                               height_ratios=[0.06, 1, 1, 1, 1, 1],
                                               hspace=0.4)
    axes2 = [fig.add_subplot(gs_rain[i]) for i in range(1, 6)]
    ax.axis("off")
    panel_label(ax, "b", outside=True,
                title="Representative features, one per group")
    for i, f in enumerate(reps):
        a, b = hc[f].dropna(), sz[f].dropna()
        aax = axes2[i]
        for pos, data, color in [(-0.32, a, HC_COLOR), (0.32, b, SZ_COLOR)]:
            vp = aax.violinplot(data, positions=[pos], widths=0.42,
                                showextrema=False)
            for pc in vp["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.35)
            aax.boxplot(data, positions=[pos], widths=0.14, showfliers=False,
                        patch_artist=True,
                        medianprops=dict(color="k", lw=0.8),
                        boxprops=dict(facecolor="white", color=color, lw=0.8),
                        whiskerprops=dict(color=color, lw=0.8),
                        capprops=dict(color=color, lw=0.8))
        p, d = welch_d(a, b)
        aax.set_title(f, fontsize=8, loc="left", pad=6)
        aax.set_xticks([-0.32, 0.32])
        aax.set_xticklabels(["HC", "SZ"], fontsize=7)
        aax.tick_params(axis="y", labelsize=6.5)
        aax.set_title(f"d={d:.2f}, p={p:.1g}", fontsize=7, loc="right", pad=6)
    # --- bottom row: (c) feature x category profiles -------------------
    feat = load_subject_features("train")
    feat = feat.join(meta["label"], on="subject_id")
    cats = feat.index.get_level_values("image").map(category_of)
    feat["category"] = cats
    figs = ["tem_dur_mean", "spa_dispersion", "geo_scanpath_len", "pup_mean"]
    subj_cat = feat.groupby(["subject_id", "category", "label"])[figs].mean().reset_index()
    gs_bot = gs[1].subgridspec(1, 4, wspace=0.3)
    axes_c = [fig.add_subplot(gs_bot[i]) for i in range(4)]
    for i, f in enumerate(figs):
        axc = axes_c[i]
        x = np.arange(4)
        for gi, (lab, color) in enumerate([(0, HC_COLOR), (1, SZ_COLOR)]):
            grp = subj_cat[subj_cat.label == lab].groupby("category")[f]
            axc.bar(x + (gi - 0.5) * 0.35, grp.mean().reindex(CAT_ORDER).values,
                    width=0.35, color=color, yerr=grp.sem().reindex(CAT_ORDER).values,
                    capsize=2.5, label="HC" if lab == 0 else "SZ")
        axc.set_xticks(x)
        axc.set_xticklabels([c[:6] for c in CAT_ORDER], fontsize=8)
        axc.set_title(f, fontsize=9.5)
        if i == 0:
            axc.set_ylabel("subject-level mean (±SEM)")
    pos = gs[1].get_position(fig)
    handles, labels = axes_c[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=2, loc="lower right",
               bbox_to_anchor=(pos.x1, pos.y1 + 0.025))
    fig.text(pos.x0, pos.y1 + 0.027, "c", fontweight="bold", fontsize=11,
             ha="left", va="bottom")
    fig.text(pos.x0 + 0.012, pos.y1 + 0.027, "Feature values depend on "
             "stimulus category", fontsize=10, color="#333333",
             ha="left", va="bottom")

    savefig(fig, OUT, "Figure_3")
    es.to_csv(TAB / "T02.02_effect_sizes.csv")
    subj_cat.groupby(["category", "label"])[figs].agg(["mean", "sem"]) \
        .to_csv(TAB / "T02.03_category_profiles.csv")


if __name__ == "__main__":
    fig_effect_sizes()
    print("done Figure_3 (effect sizes)")
