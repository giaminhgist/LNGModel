"""Correctness checks (Phase 1) — split isolation, fit-on-train-only, NaN
guards, checkpoint restore, batch-1 BN, mahal dimension.

Usage: python src/checks/check_correctness.py  (from repo root)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data.common import official_folds, load_feature_names, labels_of  # noqa: E402
from data.tabular import hc_normative_stats, feature_norm_stats  # noqa: E402
from models.proposal import NormativeModel, NormativeDataset  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from seeding import seed_all  # noqa: E402

FEAT = load_feature_names()
checks = []


def check(name, cond, detail=""):
    checks.append((name, bool(cond), detail))


def main():
    folds = list(official_folds())

    # 1. split isolation: disjoint, sizes, HC/SZ on both sides
    for fold_name, train_ids, val_ids in folds:
        check(f"1a {fold_name} disjoint",
              set(train_ids).isdisjoint(val_ids),
              f"{len(train_ids)}/{len(val_ids)}")
        check(f"1b {fold_name} sizes 120/40",
              len(train_ids) == 120 and len(val_ids) == 40)
        y_tr, y_va = labels_of(train_ids), labels_of(val_ids)
        check(f"1c {fold_name} both classes in train",
              len(np.unique(y_tr)) == 2)
        check(f"1d {fold_name} both classes in val",
              len(np.unique(y_va)) == 2,
              f"val HC={int((y_va == 0).sum())} SZ={int((y_va == 1).sum())}")

    fold_name, train_ids, val_ids = folds[0]

    # 2. stats fit ONLY on the passed ids (sensitivity + guard on the real call)
    mu_a, sig_a = feature_norm_stats(train_ids)
    mu_b, sig_b = feature_norm_stats(train_ids + val_ids[:1])
    check("2a feature stats depend only on passed ids (sensitivity)",
          not np.allclose(mu_a, mu_b),
          f"mean |dmu|={np.abs(mu_a - mu_b).mean():.3g}")
    hc_val = [s for s in val_ids if s < 200][:1]
    m_a, s_a, ss_a = hc_normative_stats(train_ids)
    m_b, s_b, ss_b = hc_normative_stats(train_ids + hc_val)
    check("2b HC stats depend only on passed ids (sensitivity)",
          not np.allclose(m_a, m_b),
          f"mean |dmu|={np.abs(m_a - m_b).mean():.3g}")
    check("2c val HC not among stats inputs by construction",
          set(hc_val).isdisjoint(train_ids))

    # 3. dataset tensors finite; mask = all-NaN stimulus rows
    ds = NormativeDataset(val_ids, train_ids, "learned", feature_cols=FEAT)
    for i in range(len(ds)):
        (X, mask), y, sid = ds[i]
        check(f"3a subject {sid} finite X/mask",
              torch.isfinite(X).all() and torch.isfinite(mask).all())
    check("3b learned D shape (40, 100, 45)", ds[0][0][0].shape == (100, 45),
          str(tuple(ds[0][0][0].shape)))
    ds_m = NormativeDataset(val_ids, train_ids, "mahal", feature_cols=FEAT)
    check("3c mahal D shape (100, 46)", ds_m[0][0][0].shape == (100, 46),
          str(tuple(ds_m[0][0][0].shape)))

    # 5. checkpoint restore reproduces predictions (2-epoch mini run)
    import subprocess
    subprocess.run([sys.executable, "src/scripts/train_proposal.py",
                    "--experiment", "CORRECTCHECK", "--ablation", "mlp_deepset",
                    "--comparator", "mlp", "--pool", "deepset",
                    "--fold", fold_name, "--seed", "5", "--epochs", "2",
                    "--feature_set", "full45"], cwd=ROOT, check=True,
                   capture_output=True, text=True)
    rd = ROOT / "outputs" / "experiments" / "CORRECTCHECK" / "full45" / "seed5" / f"fold{fold_name}"
    summ = json.loads((rd / "summary.json").read_text())
    ckpt = torch.load(rd / "best.pt", weights_only=False, map_location="cpu")
    seed_all(5)
    model2 = NormativeModel(d_in=45, comparator="mlp", pool="deepset",
                            dropout=0.3)
    model2.load_state_dict(ckpt["model_state"])
    model2.eval()
    val_ds = NormativeDataset(val_ids, train_ids, "learned", feature_cols=FEAT)
    probs = []
    with torch.no_grad():
        for i in range(len(val_ds)):
            xs = val_ds.collate([val_ds[i][0]])
            p, _ = model2(xs)
            probs.append(float(p.item()))
    m = compute_metrics(labels_of(val_ids), np.array(probs))
    check("5a restored best.pt reproduces AUC",
          abs(m["auc"] - summ["best_val_metrics"]["auc"]) < 1e-12,
          f"{m['auc']:.12f} vs {summ['best_val_metrics']['auc']:.12f}")
    check("5b restored probs equal saved probs",
          np.allclose(np.array(ckpt["val_probs"]), np.array(probs)))

    # 6. batch-1 BatchNorm path (FNN eval with a single subject)
    from models.baselines import BasicFNN
    seed_all(0)
    fnn = BasicFNN(in_dim=91).eval()
    with torch.no_grad():
        p, _ = fnn(torch.randn(1, 91))
    check("6 FNN eval with batch size 1 (BN running stats)",
          torch.isfinite(p).all())

    # 7. classical seed threading (svm_rbf differs, gnb identical)
    from features.subject import build_agg
    from models.baselines import make_ml_pipeline
    from data.common import load_metadata
    X = build_agg(sorted(train_ids + val_ids))
    meta = load_metadata()
    Xtr = X.loc[sorted(train_ids)].values
    ytr = meta.loc[sorted(train_ids), "label"].to_numpy()
    rng = np.random.RandomState(0)
    i_tr = rng.choice(len(train_ids), 40, replace=False)
    p0 = make_ml_pipeline("svm_rbf", 0).fit(Xtr[i_tr], ytr[i_tr]).predict_proba(Xtr[i_tr])[:, 1]
    p1 = make_ml_pipeline("svm_rbf", 1).fit(Xtr[i_tr], ytr[i_tr]).predict_proba(Xtr[i_tr])[:, 1]
    check("7a svm_rbf stochastic per seed", not np.allclose(p0, p1))
    g0 = make_ml_pipeline("gnb", 0).fit(Xtr[i_tr], ytr[i_tr]).predict_proba(Xtr[i_tr])[:, 1]
    g1 = make_ml_pipeline("gnb", 1).fit(Xtr[i_tr], ytr[i_tr]).predict_proba(Xtr[i_tr])[:, 1]
    check("7b gnb deterministic across seeds (documented)", np.allclose(g0, g1))

    print(f"{'check':<58} {'pass':<6} detail")
    ok = True
    for name, passed, detail in checks:
        print(f"{name:<58} {'OK' if passed else 'FAIL':<6} {detail}")
        ok &= passed
    # remove the smoke-run dir (smoke runs are not part of the report)
    import shutil
    d = ROOT / "outputs" / "experiments" / "CORRECTCHECK"
    if d.exists():
        shutil.rmtree(d)
    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
