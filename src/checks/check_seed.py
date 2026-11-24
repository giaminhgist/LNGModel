"""Seed reproducibility checks (Phase 1, minimum meaningful set).

1. Two fresh processes, same config/seed/fold -> identical initial parameter
   hash AND identical short-run metrics (published tolerance: exact match on
   CPU — same code, same seed, same device; no CPU/GPU bitwise promise).
2. Two different seeds -> different neural initial parameters.
3. One fold run alone == the same fold inside `--fold all` (init + metrics):
   folds do not depend on the previous fold's RNG state.
4. The CLI seed reaches sklearn estimators with randomness
   (svm_rbf/rf/lr_l1 differ across seeds; gnb/knn stay identical — valid for
   deterministic methods).

Usage: python src/checks/check_seed.py  (from repo root)
Smoke runs use --epochs 2 and are NOT part of the experiment report.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def sh(args, cwd=ROOT):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                          check=True).stdout.strip()


def init_hash(seed, **kw):
    cmd = [PY, "src/checks/_init_hash.py", "--seed", str(seed)]
    for k, v in kw.items():
        cmd += [f"--{k}", str(v)]
    return sh(cmd)


def short_run(experiment, fold, seed, epochs=2, read_fold=None):
    """Run the mlp_deepset config for `epochs` epochs; return metrics.jsonl rows
    with the wall-clock field `elapsed_s` stripped (run-specific metadata,
    not a scientific result)."""
    cmd = [PY, "src/scripts/train_proposal.py", "--experiment", experiment,
           "--ablation", "mlp_deepset", "--comparator", "mlp",
           "--pool", "deepset", "--fold", fold, "--seed", str(seed),
           "--epochs", str(epochs), "--feature_set", "full45"]
    sh(cmd)
    d = (ROOT / "outputs" / "experiments" / experiment / "full45" /
         f"seed{seed}" / f"fold{read_fold or fold}")
    rows = [json.loads(ln) for ln in (d / "metrics.jsonl").read_text().splitlines()]
    return [{k: v for k, v in r.items() if k != "elapsed_s"} for r in rows]


def main():
    checks = []
    base = dict(deviation="learned", comparator="mlp", pool="deepset")

    # 1a. same seed -> identical init hash across two fresh processes
    h1 = init_hash(42, **base)
    h2 = init_hash(42, **base)
    checks.append(("1a init hash deterministic across processes", h1 == h2, h1[:16]))

    # 1b. same seed -> identical short-run metrics across two fresh processes
    m1 = short_run("SEEDCHECK-A", "Set_1", 42)
    m2 = short_run("SEEDCHECK-B", "Set_1", 42)
    checks.append(("1b short-run metrics identical across processes",
                   m1 == m2, f"{len(m1)} epochs"))

    # 2. different seeds -> different init
    h3 = init_hash(7, **base)
    checks.append(("2 different seeds -> different init", h1 != h3,
                   f"{h1[:12]} vs {h3[:12]}"))

    # 3. fold-alone vs fold inside --fold all
    m_alone = short_run("SEEDCHECK-ALONE", "Set_1", 7)
    m_seq = short_run("SEEDCHECK-SEQ", "all", 7, read_fold="Set_1")
    # run_info of the sequential run's Set_1 fold must match the alone run's
    ri_alone = json.loads((ROOT / "outputs" / "experiments" / "SEEDCHECK-ALONE" /
                           "full45" / "seed7" / "foldSet_1" /
                           "run_info.json").read_text())
    ri_seq = json.loads((ROOT / "outputs" / "experiments" / "SEEDCHECK-SEQ" /
                         "full45" / "seed7" / "foldSet_1" /
                         "run_info.json").read_text())
    same_ids = ri_alone["train_subject_ids"] == ri_seq["train_subject_ids"]
    checks.append(("3a fold-alone == fold-in-all (train/val ids)",
                   same_ids, f"{len(ri_alone['train_subject_ids'])} subjects"))
    checks.append(("3b fold-alone == fold-in-all (metrics)",
                   m_alone == m_seq, f"{len(m_alone)} epochs"))

    # 4. CLI seed reaches sklearn estimators
    sys.path.insert(0, str(ROOT / "src"))
    from features.subject import build_agg  # noqa: E402
    from models.baselines import make_ml_pipeline  # noqa: E402
    from data.common import official_folds, load_metadata  # noqa: E402
    import numpy as np

    fold_name, train_ids, val_ids = next(iter(official_folds()))
    X = build_agg(sorted(train_ids + val_ids))
    meta = load_metadata()
    Xtr = X.loc[sorted(train_ids)].values
    ytr = meta.loc[sorted(train_ids), "label"].to_numpy()
    rng = np.random.RandomState(0)
    i_tr = rng.choice(len(train_ids), 40, replace=False)
    p0 = make_ml_pipeline("svm_rbf", 0).fit(Xtr[i_tr], ytr[i_tr]) \
        .predict_proba(Xtr[i_tr])[:, 1]
    p1 = make_ml_pipeline("svm_rbf", 1).fit(Xtr[i_tr], ytr[i_tr]) \
        .predict_proba(Xtr[i_tr])[:, 1]
    g0 = make_ml_pipeline("gnb", 0).fit(Xtr[i_tr], ytr[i_tr]) \
        .predict_proba(Xtr[i_tr])[:, 1]
    g1 = make_ml_pipeline("gnb", 1).fit(Xtr[i_tr], ytr[i_tr]) \
        .predict_proba(Xtr[i_tr])[:, 1]
    checks.append(("4a seed reaches svm_rbf (0 != 1)", not np.allclose(p0, p1),
                   f"max|d|={np.abs(p0 - p1).max():.3g}"))
    checks.append(("4b gnb deterministic across seeds (valid)",
                   np.allclose(g0, g1), "identical"))

    print(f"{'check':<52} {'pass':<6} detail")
    ok = True
    for name, passed, detail in checks:
        print(f"{name:<52} {'OK' if passed else 'FAIL':<6} {detail}")
        ok &= bool(passed)
    # remove smoke-run dirs (smoke runs are not part of the report)
    import shutil
    for name in ("SEEDCHECK-A", "SEEDCHECK-B", "SEEDCHECK-ALONE",
                 "SEEDCHECK-SEQ"):
        d = ROOT / "outputs" / "experiments" / name
        if d.exists():
            shutil.rmtree(d)
    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
