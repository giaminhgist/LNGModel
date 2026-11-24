"""Unified evaluation metrics — single implementation for proposal and baselines.

Convention (EMS benchmark): SZ = positive class (label 1), HC = negative
(label 0). ACC/SEN/SPEC/F1 are computed at a FIXED threshold 0.5 on the
sigmoid/score outputs; AUC is threshold-free on the continuous P(SZ).

Keys: acc, auc, balanced_acc, sen, spec, f1, pre, tp, tn, fp, fn, threshold.
`spec` replaces the legacy `spe` key used by the old baseline code; readers of
old artifacts should map `spe` -> `spec`.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

THRESHOLD = 0.5
METRIC_KEYS = ["acc", "auc", "balanced_acc", "sen", "spec", "f1"]


def compute_metrics(y_true, y_score, threshold=THRESHOLD):
    """Return dict of the 6 headline metrics + confusion counts at `threshold`.

    AUC is NaN only when the labels contain fewer than 2 classes (e.g. a fold
    without HC or without SZ); callers must record the reason, never silently
    replace it with 0.5.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    sen = tp / max(tp + fn, 1)          # sensitivity / recall
    spec = tn / max(tn + fp, 1)         # specificity
    bal = (sen + spec) / 2.0            # balanced accuracy
    pre = tp / max(tp + fp, 1)
    f1 = 2 * pre * sen / max(sen + pre, 1e-12)
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_score))
        auc_reason = None
    else:
        auc = float("nan")
        auc_reason = f"single class in labels ({int(np.unique(y_true)[0])})"
    return {"acc": acc, "auc": auc, "balanced_acc": bal, "sen": sen,
            "spec": spec, "pre": pre, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "threshold": float(threshold), "auc_reason": auc_reason}


def mean_std(list_of_metrics, ddof=1, keys=METRIC_KEYS):
    """Mean ± std (sample, ddof=1) over a list of metric dicts.

    NaN entries are excluded per key; returns {key: (mean, std)} or
    {key: (nan, nan)} when no non-NaN value exists.
    """
    out = {}
    for key in keys:
        vals = [m[key] for m in list_of_metrics if key in m and not np.isnan(m[key])]
        out[key] = (float(np.mean(vals)), float(np.std(vals, ddof=ddof))) \
            if vals else (float("nan"), float("nan"))
    return out


def summarize(list_of_metrics, ddof=1, keys=METRIC_KEYS):
    """Dict form of mean_std: {key: {"mean": m, "std": s}}."""
    out = {}
    for key, (m, s) in mean_std(list_of_metrics, ddof=ddof, keys=keys).items():
        out[key] = {"mean": m, "std": s}
    return out
