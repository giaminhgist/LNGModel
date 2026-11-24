"""Experiment matrix runner: 5 proposal configs + 10 baselines x 10 locked
seeds x official 4-fold, in parallel subprocess workers.

One subprocess per (experiment, seed) [proposal] or (method, seed) [baseline];
each subprocess runs the 4 folds sequentially (folds are RNG-independent
because the trainer reseeds at the start of every fold).

Results are stored under outputs/experiments/ (NOT "phase3").

Usage (from repo root):
    python src/scripts/run_experiments.py --workers 8     # run all pending jobs
    python src/scripts/run_experiments.py --only baselines
    python src/scripts/run_experiments.py --only proposal
    python src/scripts/run_experiments.py --retry-failed
    python src/scripts/run_experiments.py --status

Resume: a job is skipped only when its artifacts are complete AND its recorded
data/code hashes match the current dataset — exit code 0 alone is never
trusted. Failed jobs are re-run with the same seed/config; seeds are never
dropped from the sample.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.common import ROOT, OUTPUTS, data_identity  # noqa: E402
from seeding import code_hash  # noqa: E402
from registry import (SEEDS, PROPOSAL_EXPERIMENTS, BASELINE_METHODS,  # noqa: E402
                      FEATURE_SET, FOLD_NAMES, PROPOSAL_HP)  # noqa: E402

BASE = OUTPUTS / "experiments"
STATUS = BASE / "run_status.csv"
LOG_DIR = BASE / "logs"


def build_jobs():
    """Full job list: proposal = (experiment, seed), baseline = (method, seed)."""
    jobs = []
    for exp in PROPOSAL_EXPERIMENTS:
        for seed in SEEDS:
            jobs.append({"kind": "proposal", "id": f"{exp['id']}-s{seed}",
                         "exp": exp, "seed": seed})
    for method, rep in BASELINE_METHODS.items():
        for seed in SEEDS:
            jobs.append({"kind": "baseline", "id": f"{method}-s{seed}",
                         "method": method, "rep": rep, "seed": seed})
    return jobs


def job_cmd(job):
    py = sys.executable
    if job["kind"] == "proposal":
        e = job["exp"]
        cmd = [py, "src/scripts/train_proposal.py",
               "--experiment", e["id"], "--ablation", e["ablation"],
               "--deviation", e["deviation"], "--pool", e["pool"],
               "--fold", "all", "--seed", str(job["seed"]),
               "--epochs", str(PROPOSAL_HP["epochs"]),
               "--patience", str(PROPOSAL_HP["patience"]),
               "--feature_set", FEATURE_SET]
        if e["comparator"] is not None:
            cmd += ["--comparator", e["comparator"]]
        return cmd
    cmd = [py, "src/scripts/run_baseline.py",
           "--method", job["method"], "--seed", str(job["seed"])]
    return cmd


# --------------------------------------------------------------------------- #
# completeness checks (resume/skip)
# --------------------------------------------------------------------------- #

def _fold_ok(fold_dir, expect_seed, expect_experiment):
    """Artifact completeness; hashes are compared against the CURRENT
    data/code state (evaluated at check time)."""
    if not (fold_dir / "summary.json").exists() or not (fold_dir / "best.pt").exists() \
            or not (fold_dir / "run_info.json").exists() or not (fold_dir / "config.json").exists():
        return False
    try:
        ri = json.loads((fold_dir / "run_info.json").read_text())
        cfg = json.loads((fold_dir / "config.json").read_text())
    except Exception:
        return False
    if ri.get("status") != "completed":
        return False
    if ri.get("data", {}).get("data_hash") != data_identity()["data_hash"]:
        return False
    if ri.get("code_hash") != code_hash():
        return False
    if cfg.get("seed") != expect_seed or cfg.get("experiment") != expect_experiment:
        return False
    return True


def _classical_fold_ok(fold_dir, expect_seed, expect_method):
    if not (fold_dir / "metrics.json").exists() or not (fold_dir / "val_preds.csv").exists():
        return False
    if not (fold_dir / "run_info.json").exists():
        return False
    try:
        ri = json.loads((fold_dir / "run_info.json").read_text())
        m = json.loads((fold_dir / "metrics.json").read_text())
    except Exception:
        return False
    if ri.get("status") != "completed":
        return False
    if ri.get("data", {}).get("data_hash") != data_identity()["data_hash"]:
        return False
    if ri.get("code_hash") != code_hash():
        return False
    if m.get("seed") != expect_seed or m.get("method") != expect_method:
        return False
    return True


def job_complete(job):
    """True only when all 4 folds have complete, hash-matching artifacts."""
    if job["kind"] == "proposal":
        base = BASE / job["exp"]["id"] / FEATURE_SET / f"seed{job['seed']}"
        if not (base / "folds_summary.json").exists():
            return False
        return all(_fold_ok(base / f"fold{f}", job["seed"], job["exp"]["id"])
                   for f in FOLD_NAMES)
    base = BASE / job["method"] / job["rep"] / f"seed{job['seed']}"
    if not (base / "folds_summary.json").exists():
        return False
    for f in FOLD_NAMES:
        d = base / f"fold{f}"
        if job["method"] in ("fnn", "fnn_cat"):
            if not _fold_ok(d, job["seed"], job["method"]):
                return False
        else:
            if not _classical_fold_ok(d, job["seed"], job["method"]):
                return False
    return True


# --------------------------------------------------------------------------- #

def run_one(job):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{job['id']}.log"
    started = datetime.now().isoformat(timespec="seconds")
    cmd = job_cmd(job)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    with open(log, "w") as f:
        f.write(f"### {job['id']} started {started}\n$ {' '.join(cmd)}\n")
        f.flush()
        t0 = datetime.now()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        dur = (datetime.now() - t0).total_seconds()
        f.write(f"\n### exit={proc.returncode} duration={dur:.0f}s\n")
    if proc.returncode != 0:
        status = "failed"
    elif job_complete(job):
        status = "completed"
    else:
        status = "stale"
    return {"id": job["id"], "status": status, "started": started,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(dur, 1), "exit": proc.returncode}


def load_status():
    if STATUS.exists():
        df = pd.read_csv(STATUS, index_col=0)
    else:
        df = pd.DataFrame(columns=["status", "started", "finished",
                                   "duration_s", "exit"])
    if df.index.name != "id":
        df.index.name = "id"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--jobs", default=None,
                    help="comma-separated job ids, e.g. EXP-PROP-001-s42")
    ap.add_argument("--only", choices=["proposal", "baselines"], default=None)
    ap.add_argument("--force", action="store_true", help="re-run completed jobs")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--status", action="store_true", help="print status only")
    args = ap.parse_args()

    jobs = build_jobs()
    if args.only == "proposal":
        jobs = [j for j in jobs if j["kind"] == "proposal"]
    elif args.only == "baselines":
        jobs = [j for j in jobs if j["kind"] == "baseline"]
    if args.jobs:
        wanted = set(args.jobs.split(","))
        jobs = [j for j in jobs if j["id"] in wanted]

    status = load_status()
    if args.status:
        if len(status):
            print(status.sort_index().to_string())
        else:
            print("no status yet")
        return

    todo = []
    for j in jobs:
        if args.force:
            todo.append(j)
        elif j["id"] in status.index and status.loc[j["id"], "status"] == "completed":
            if args.retry_failed or not job_complete(j):
                todo.append(j)
        elif j["id"] in status.index and status.loc[j["id"], "status"] == "failed" \
                and not args.retry_failed:
            continue
        elif j["id"] in status.index and status.loc[j["id"], "status"] == "stale":
            todo.append(j)
        elif j["id"] not in status.index and job_complete(j):
            status.loc[j["id"]] = {"status": "completed", "started": "",
                                   "finished": "", "duration_s": 0, "exit": 0}
            status.to_csv(STATUS)
        else:
            todo.append(j)

    print(f"{len(todo)}/{len(jobs)} jobs to run with {args.workers} workers",
          flush=True)
    if not todo:
        return

    pool = ThreadPoolExecutor(max_workers=args.workers)
    wave_size = max(args.workers * 2, 1)
    for start in range(0, len(todo), wave_size):
        chunk = todo[start:start + wave_size]
        futures = {pool.submit(run_one, j): j for j in chunk}
        while futures:
            done, pending = wait(list(futures), timeout=900,
                                 return_when=FIRST_COMPLETED)
            if not done:
                print(f"  [watchdog] {len(pending)} jobs still running",
                      flush=True)
                continue
            for fut in done:
                job = futures.pop(fut)
                try:
                    row = fut.result()
                except Exception as exc:
                    row = {"id": job["id"], "status": "failed",
                           "started": "", "finished": "", "duration_s": 0,
                           "exit": -1}
                    (LOG_DIR / f"{job['id']}.log").write_text(
                        f"### {job['id']} crashed\n{exc}\n")
                jid = row.pop("id")
                status.loc[jid] = row
                status.to_csv(STATUS)
                print(f"[{row['status']:9s}] {jid}  ({row['duration_s']:.0f}s)",
                      flush=True)
        print(f"  wave {start // wave_size + 1} done", flush=True)
    pool.shutdown(wait=True)
    print(f"status -> {STATUS}", flush=True)


if __name__ == "__main__":
    main()
