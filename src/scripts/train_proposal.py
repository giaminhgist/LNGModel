"""Train the main proposal (Learned Stimulus-Conditioned Normative Modeling).

Usage (from repo root):
    python src/scripts/train_proposal.py --experiment EXP-PROP-001 --ablation mlp_deepset \
        --comparator mlp --pool deepset --fold all --seed 42
    python src/scripts/train_proposal.py --experiment EXP-PROP-004 --ablation z_mean \
        --deviation z --pool mean --fold all --seed 42
    python src/scripts/train_proposal.py --experiment EXP-PROP-005 --ablation mlp_max \
        --comparator mlp --pool max --fold all --seed 42

Per fold, a run directory is created:
    outputs/experiments/{experiment}/{feature_set}/seed{seed}/fold{fold}/
with config.json, run_info.json, metrics.jsonl, val_probs.jsonl, best.pt,
last.pt, predictions.csv, summary.json.
A per-seed summary over the 4 folds is written to
    outputs/experiments/{experiment}/{feature_set}/seed{seed}/folds_summary.json
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # EMS-Projects/src
from data.common import official_folds, load_feature_names  # noqa: E402
from training.config import parse_args  # noqa: E402
from training.trainer import train_model  # noqa: E402
from metrics import summarize  # noqa: E402
from models.proposal import NormativeModel, NormativeDataset  # noqa: E402


def main():
    cfg, device_arg = parse_args(default_proposal="proposal")
    device = device_arg or ("cuda" if torch.cuda.is_available() else "cpu")
    deviation = cfg.extra["deviation"]
    comparator = cfg.extra["comparator"]
    pool = cfg.extra["pool"]

    feature_cols = load_feature_names()  # full45 schema (Phase 3)
    n_stim = 100

    folds = list(official_folds()) if cfg.fold == "all" else [
        (cfg.fold, *next(f for f in official_folds() if f[0] == cfg.fold)[1:])]
    fold_metrics = []
    for fold_name, train_ids, val_ids in folds:
        cfg.fold = fold_name  # per-fold run dirs get the real fold name
        train_ds = NormativeDataset(train_ids, train_ids, deviation,
                                    feature_cols=feature_cols)
        val_ds = NormativeDataset(val_ids, train_ids, deviation,
                                  feature_cols=feature_cols)
        # model factory: the trainer seeds BEFORE calling it, so the initial
        # parameters depend only on cfg.seed — not on any previous fold's
        # RNG state.
        factory = (lambda: NormativeModel(
            d_in=len(feature_cols), comparator=comparator, pool=pool,
            dropout=cfg.dropout, deviation=deviation, n_stim=n_stim))
        summary = train_model(factory, train_ds, val_ds, cfg, device)
        fold_metrics.append({**summary["best_val_metrics"], "fold": fold_name,
                             "best_epoch": summary["best_epoch"],
                             "epochs_trained": summary["epochs_trained"],
                             "stopping_reason": summary["stopping_reason"]})
        print(f"{fold_name}: AUC={summary['best_val_metrics']['auc']:.4f} "
              f"Acc={summary['best_val_metrics']['acc']:.4f} "
              f"Sen={summary['best_val_metrics']['sen']:.4f} "
              f"Spec={summary['best_val_metrics']['spec']:.4f} "
              f"F1={summary['best_val_metrics']['f1']:.4f} "
              f"(best epoch {summary['best_epoch']}, "
              f"{summary['epochs_trained']} epochs, "
              f"{summary['stopping_reason']})")

    agg = summarize(fold_metrics)
    out = {"experiment": cfg.experiment, "ablation": cfg.ablation,
           "feature_set": cfg.feature_set, "seed": cfg.seed,
           "folds": [m["fold"] for m in fold_metrics],
           "fold_metrics": fold_metrics, "mean_metrics": agg}
    if len(folds) > 1:
        base = cfg.run_dir(create=True).parent
        (base / "folds_summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
