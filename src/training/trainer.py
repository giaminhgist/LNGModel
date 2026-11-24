"""Generic supervised trainer for the normative-gaze proposal and FNN baselines.

Responsibilities:
  - seeds BEFORE model construction (model passed as a factory), reseeding at
    the start of every fold so folds are independent of run order
  - train/val loops with per-epoch metrics appended (and flushed) to
    metrics.jsonl
  - train metrics computed at eval() after each epoch over ALL train subjects
    (no dropout, same model/bank as training); per-batch online loss logged
    as train_online.loss
  - early stopping on validation AUC (patience, earliest epoch wins ties),
    best checkpoint to best.pt, final epoch to last.pt (both include the
    normative bank buffers)
  - per-epoch validation subject probabilities to val_probs.jsonl
  - run_info.json (environment, git commit, data/schema hashes, split ids)
  - returns the run summary (best epoch metrics + all epochs)

Dataset contract (implemented per model):
  ds.subjects      -> list of subject ids
  ds.label_of(sid) -> 0/1  (optional, for info)
  ds.__getitem__(idx) -> (x, y, sid) with x model-specific
  ds.collate(list_of_x) -> batched x
Model contract:
  model(x) -> (prob (B,1), aux dict); aux may contain "emb" (B,D)
  model.on_epoch_start()            # optional hook (bank refresh)
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from metrics import compute_metrics
from seeding import seed_all, determinism_env_info, code_hash

HEADLINE = ["acc", "auc", "balanced_acc", "sen", "spec", "f1"]


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[3], check=False
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return [_to_device(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


@torch.no_grad()
def _evaluate(model, ds, device, loss_fn, eval_batch_size=64):
    """Batch evaluation in eval() mode. Returns
    (metrics, ys, probs, sids, embs_or_None, mean_bce_loss)."""
    model.eval()
    ys, probs, sids, embs = [], [], [], []
    loss_sum, n = 0.0, 0
    for i in range(0, len(ds), eval_batch_size):
        batch = list(range(i, min(i + eval_batch_size, len(ds))))
        items = [ds[j] for j in batch]
        xs = _to_device(ds.collate([it[0] for it in items]), device)
        yb = torch.tensor([it[1] for it in items], dtype=torch.float32,
                          device=device).view(-1, 1)
        prob, aux = model(xs)
        loss_sum += loss_fn(prob.view(-1), yb.view(-1)).item() * len(items)
        n += len(items)
        ys += [it[1] for it in items]
        sids += [it[2] for it in items]
        probs += [float(p) for p in prob.view(-1)]
        if aux.get("emb") is not None:
            embs.append(aux["emb"].cpu().numpy())
    m = compute_metrics(np.array(ys), np.array(probs))
    return (m, np.array(ys), np.array(probs), sids,
            (np.concatenate(embs, 0) if embs else None),
            loss_sum / max(n, 1))


def _write_val_probs(path, ys, probs, sids, epoch, cfg):
    """Append per-epoch val subject probabilities (jsonl, flushed)."""
    with open(path, "a") as f:
        for y, p, sid in zip(ys, probs, sids):
            f.write(json.dumps({
                "subject_id": int(sid), "label": int(y), "prob": p,
                "epoch": epoch, "seed": cfg.seed, "fold": cfg.fold,
                "experiment": cfg.experiment, "feature_set": cfg.feature_set,
            }) + "\n")
        f.flush()


def train_model(model_factory, train_ds, val_ds, cfg, device, run_info=None):
    """Full training loop. Returns dict with run summary.

    model_factory() is called AFTER seed_all(cfg.seed), so initial parameters
    depend only on (cfg.seed), not on any prior fold's RNG state.
    """
    # ---- seed FIRST, then build everything random ----
    seed_all(cfg.seed)
    model = model_factory()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCELoss()
    run_dir = cfg.run_dir(create=True)

    from data.common import data_identity
    info = {
        "git_commit": _git_commit(),
        "code_hash": code_hash(),
        "env": determinism_env_info(),
        "data": data_identity(),
        "train_subject_ids": [int(s) for s in train_ds.subjects],
        "val_subject_ids": [int(s) for s in val_ds.subjects],
        "n_train": len(train_ds.subjects), "n_val": len(val_ds.subjects),
        "device": device, "status": "failed", "started": None,
        "finished": None,
    }
    info.update(run_info or {})
    assert len(set(info["train_subject_ids"]) & set(info["val_subject_ids"])) == 0, \
        "train/val subject overlap"
    cfg.save(run_dir)
    info["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))

    log_path = run_dir / "metrics.jsonl"
    val_probs_path = run_dir / "val_probs.jsonl"
    log_path.write_text("")      # truncate: a re-run must start clean
    val_probs_path.write_text("")

    best_auc, best_state, best_epoch, wait = -1.0, None, -1, 0
    best_val_metrics, best_val_ys, best_val_probs, best_val_sids = None, None, None, None
    history = []
    stopping_reason = "max_epochs"
    for epoch in range(cfg.epochs):
        t0 = time.time()
        # ---- train ----
        model.train()
        if hasattr(model, "refresh_bank"):      # rebuild the HC latent bank
            model.refresh_bank(train_ds)        # (lagged-by-epoch schedule)
        order = np.random.permutation(len(train_ds))
        on_loss, on_n = 0.0, 0
        for i in range(0, len(order), cfg.batch_size):
            batch = order[i:i + cfg.batch_size]
            items = [train_ds[j] for j in batch]
            xs = _to_device(train_ds.collate([it[0] for it in items]), device)
            yb = torch.tensor([it[1] for it in items], dtype=torch.float32,
                              device=device).view(-1, 1)
            opt.zero_grad()
            prob, aux = model(xs, y=yb)
            loss = loss_fn(prob, yb)
            assert torch.isfinite(loss), f"non-finite loss at epoch {epoch}"
            loss.backward()
            opt.step()
            on_loss += loss.item() * len(batch)
            on_n += len(batch)

        # ---- epoch-end train metrics at eval() on ALL train subjects ----
        tr_metrics, _, _, _, _, tr_loss = _evaluate(model, train_ds, device, loss_fn)

        # ---- val ----
        val_metrics, val_ys, val_probs, val_sids, _, val_loss = \
            _evaluate(model, val_ds, device, loss_fn)

        auc = val_metrics["auc"]
        if np.isnan(auc):
            raise RuntimeError(f"val AUC is NaN at epoch {epoch} "
                               f"(reason: {val_metrics['auc_reason']}) — abort run")
        best_so_far = False
        if auc > best_auc:
            best_auc, best_epoch, wait = auc, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_val_metrics, best_val_ys = val_metrics, val_ys
            best_val_probs, best_val_sids = val_probs, val_sids
            best_so_far = True
        else:
            wait += 1
            if wait >= cfg.patience:
                stopping_reason = "early_stopping"

        lr = float(opt.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "lr": lr,
            "elapsed_s": round(time.time() - t0, 3),
            "best_so_far": best_so_far,
            "train": {**{k: tr_metrics[k] for k in HEADLINE},
                      "tp": tr_metrics["tp"], "tn": tr_metrics["tn"],
                      "fp": tr_metrics["fp"], "fn": tr_metrics["fn"],
                      "loss": tr_loss},
            "train_online": {
                "loss": on_loss / max(on_n, 1)},
            "val": {**{k: val_metrics[k] for k in HEADLINE},
                    "tp": val_metrics["tp"], "tn": val_metrics["tn"],
                    "fp": val_metrics["fp"], "fn": val_metrics["fn"],
                    "loss": val_loss},
            "threshold": val_metrics["threshold"],
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
        _write_val_probs(val_probs_path, val_ys, val_probs, val_sids, epoch, cfg)
        history.append(row)
        if stopping_reason == "early_stopping":
            break
            wait += 1
            if wait >= cfg.patience:
                stopping_reason = "early_stopping"
                break

    epochs_trained = epoch + 1

    # ---- last checkpoint (state at the end of the final executed epoch) ----
    last_val_metrics, last_val_ys, last_val_probs, last_val_sids, _, last_loss = \
        _evaluate(model, val_ds, device, loss_fn)
    torch.save({"model_state": {k: v.clone() for k, v in model.state_dict().items()},
                "epoch": epoch, "config": cfg.to_dict()},
               run_dir / "last.pt")

    # ---- restore best + verify its predictions reproduce exactly ----
    model.load_state_dict(best_state)
    verify_metrics, verify_ys, verify_probs, verify_sids, _, _ = \
        _evaluate(model, val_ds, device, loss_fn)
    assert abs(verify_metrics["auc"] - best_auc) < 1e-9 and \
        np.allclose(np.asarray(verify_probs, dtype=np.float64),
                    np.asarray(best_val_probs, dtype=np.float64)), \
        "best-checkpoint restore does not reproduce its validation predictions"
    torch.save({"model_state": best_state, "epoch": best_epoch,
                "val_auc": best_auc, "val_metrics": best_val_metrics,
                "val_subject_ids": [int(s) for s in best_val_sids],
                "val_probs": [float(p) for p in best_val_probs],
                "config": cfg.to_dict()}, run_dir / "best.pt")

    import pandas as pd
    pd.DataFrame({"subject_id": [int(s) for s in best_val_sids],
                  "label": best_val_ys, "prob": best_val_probs}) \
        .to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame({"subject_id": [int(s) for s in last_val_sids],
                  "label": last_val_ys, "prob": last_val_probs}) \
        .to_csv(run_dir / "predictions_last.csv", index=False)

    summary = {
        "run_dir": str(run_dir), "best_epoch": best_epoch,
        "epochs_trained": epochs_trained, "stopping_reason": stopping_reason,
        "best_val_metrics": best_val_metrics,
        "last_val_metrics": last_val_metrics,
        "best_val_auc": best_auc,
        "train_subject_ids": info["train_subject_ids"],
        "val_subject_ids": info["val_subject_ids"],
        "history": history,
    }
    (run_dir / "summary.json").write_text(
        json.dumps({k: v for k, v in summary.items() if k != "history"},
                   indent=2, default=str))
    info["status"] = "completed"
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    return summary
