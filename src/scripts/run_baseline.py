"""Run one baseline method under the Phase-3 development protocol.

Usage (from repo root):
    python src/scripts/run_baseline.py --method svm_rbf --seed 42
    python src/scripts/run_baseline.py --method fnn --seed 42

Protocol: official 4-fold validation (Set_0..Set_3, 120 train / 40 validation
subjects per fold). Unified development checkpoint policy:
  - FNN: trained by the shared generic trainer; best checkpoint = best
    outer-validation AUC (earliest epoch wins ties); per-epoch train/val
    metrics in metrics.jsonl; imputer/scaler fit on the 120 train subjects
    only. NOTE: the FNN's legacy inner early-stopping split is NOT used in
    this development table (unified protocol).
  - Classical ML: one fit on the 120 train subjects; train + val metrics
    recorded with stage=fit_complete, epoch=null.

All metrics: threshold 0.5, SZ=1. Outputs:
    outputs/experiments/{method}/{rep}/seed{seed}/fold{fold}/...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # EMS-Projects/src
from config import RESULTS  # noqa: E402
from data.common import official_folds, load_metadata  # noqa: E402
from features.subject import (build_agg, build_catagg,  # noqa: E402
                              build_concat)
from metrics import compute_metrics, METRIC_KEYS  # noqa: E402
from models.baselines import fit_predict_ml, BasicFNN  # noqa: E402
from seeding import seed_all, determinism_env_info  # noqa: E402
from training.trainer import train_model  # noqa: E402
from training.config import RunConfig, PROTOCOL_VERSION  # noqa: E402
from registry import BASELINE_METHODS, FNN_HP  # noqa: E402

METHOD_REP = BASELINE_METHODS


def build_matrix(rep, subject_ids, partition="train"):
    if rep == "agg":
        return build_agg(subject_ids, partition)
    if rep == "catagg":
        return build_catagg(subject_ids, partition)
    if rep == "concat":
        return build_concat(subject_ids, partition)
    raise ValueError(rep)


class VectorDataset:
    """Subject-level vector dataset for the generic trainer (FNN path)."""

    def __init__(self, X, y, subject_ids):
        self.X = torch.tensor(np.asarray(X, dtype=np.float32))
        self.y = np.asarray(y, dtype=np.int64)
        self.subjects = [int(s) for s in subject_ids]

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        return self.X[idx], int(self.y[idx]), self.subjects[idx]

    def collate(self, xs):
        return torch.stack(xs, 0)


def _write_classical(out_dir, m, ys, probs, sids, d, train_ids, val_ids):
    m["stage"] = "fit_complete"
    m["epoch"] = None
    m["method"] = d["method"]
    m["rep"] = d["rep"]
    m["seed"] = d["seed"]
    m["fold"] = d["fold"]
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    pd.DataFrame({"subject_id": [int(s) for s in sids], "label": ys,
                  "prob": probs}).to_csv(out_dir / "val_preds.csv", index=False)
    from data.common import data_identity
    from seeding import code_hash
    import subprocess
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=False).stdout.strip() or "unknown"
    except Exception:
        git = "unknown"
    (out_dir / "run_info.json").write_text(json.dumps({
        "status": "completed",
        "git_commit": git, "code_hash": code_hash(),
        "env": determinism_env_info(), "data": data_identity(),
        "train_subject_ids": [int(s) for s in train_ids],
        "val_subject_ids": [int(s) for s in val_ids],
    }, indent=2, default=str))


def run_fold(method, rep, fold_name, train_ids, val_ids, seed, out_dir):
    meta = load_metadata()
    assert set(train_ids).isdisjoint(val_ids), "train/val subject overlap"
    X = build_matrix(rep, sorted(train_ids + val_ids))
    y = meta.loc[sorted(train_ids + val_ids), "label"]
    X_train = X.loc[sorted(train_ids)].values
    X_val = X.loc[sorted(val_ids)].values
    y_train = y.loc[sorted(train_ids)].to_numpy()
    y_val = y.loc[sorted(val_ids)].to_numpy()

    if method in ("fnn", "fnn_cat"):
        # imputer/scaler fit on the 120 train subjects only, shared transform
        seed_all(seed)
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        X_train_t = sc.fit_transform(imp.fit_transform(X_train)).astype(np.float32)
        X_val_t = sc.transform(imp.transform(X_val)).astype(np.float32)
        train_ds = VectorDataset(X_train_t, y_train, train_ids)
        val_ds = VectorDataset(X_val_t, y_val, val_ids)
        cfg = RunConfig(experiment=method, feature_set=rep,
                        protocol_version=PROTOCOL_VERSION,
                        proposal="baseline", ablation=method, seed=seed,
                        fold=fold_name, epochs=FNN_HP["epochs"],
                        patience=FNN_HP["patience"], lr=FNN_HP["lr"],
                        weight_decay=FNN_HP["weight_decay"],
                        batch_size=FNN_HP["batch_size"],
                        dropout=FNN_HP["dropout"])
        summary = train_model(lambda: BasicFNN(in_dim=X_train_t.shape[1],
                                               dropout=FNN_HP["dropout"]),
                              train_ds, val_ds, cfg, "cpu")
        return {**summary["best_val_metrics"], "fold": fold_name,
                "best_epoch": summary["best_epoch"],
                "epochs_trained": summary["epochs_trained"],
                "stopping_reason": summary["stopping_reason"]}

    # classical ML: single fit on train, evaluate train + val at threshold 0.5
    pipe, pv_train, pv_val = fit_predict_ml(method, X_train, y_train, X_val, seed)
    m_train = compute_metrics(y_train, pv_train)
    m_val = compute_metrics(y_val, pv_val)
    record = {
        "train": {k: m_train[k] for k in METRIC_KEYS},
        "val": {k: m_val[k] for k in METRIC_KEYS},
        "train_counts": {"tp": m_train["tp"], "tn": m_train["tn"],
                         "fp": m_train["fp"], "fn": m_train["fn"]},
        "val_counts": {"tp": m_val["tp"], "tn": m_val["tn"],
                       "fp": m_val["fp"], "fn": m_val["fn"]},
        "train_loss": None, "val_loss": None,  # no neural loss for classical
    }
    _write_classical(out_dir, record, y_val, pv_val, val_ids,
                     {"method": method, "rep": rep, "seed": seed,
                      "fold": fold_name}, train_ids, val_ids)
    return {**m_val, "fold": fold_name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    method = args.method
    rep = METHOD_REP[method]

    fold_metrics = []
    for fold_name, train_ids, val_ids in official_folds():
        out_dir = RESULTS / method / rep / f"seed{args.seed}" / f"fold{fold_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        m = run_fold(method, rep, fold_name, train_ids, val_ids, args.seed, out_dir)
        fold_metrics.append(m)
        print(f"  {fold_name}: auc={m['auc']:.4f} acc={m['acc']:.4f} "
              f"sen={m['sen']:.4f} spec={m['spec']:.4f} f1={m['f1']:.4f}")

    from metrics import summarize
    agg = summarize(fold_metrics)
    out = {"method": method, "rep": rep, "seed": args.seed,
           "folds": [m["fold"] for m in fold_metrics],
           "fold_metrics": fold_metrics, "mean_metrics": agg,
           "env": determinism_env_info()}
    (RESULTS / method / rep / f"seed{args.seed}" / "folds_summary.json") \
        .write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
