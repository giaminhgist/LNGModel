"""Helper: print the sha256 of a freshly-seeded model's initial parameters.

Used by check_seed.py to compare initializations across processes. Replicates
the exact construction path of src/scripts/train_proposal.py: seed_all(seed)
THEN model construction, same dataset/stats code path.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # EMS-Projects/src
from seeding import seed_all  # noqa: E402
from models.proposal import NormativeModel  # noqa: E402
from data.common import load_feature_names  # noqa: E402


def param_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--deviation", default="learned")
    ap.add_argument("--comparator", default="mlp")
    ap.add_argument("--pool", default="attention")
    args = ap.parse_args()
    n_feat = len(load_feature_names())
    seed_all(args.seed)  # BEFORE model construction — the audited property
    model = NormativeModel(d_in=n_feat, comparator=args.comparator,
                           pool=args.pool, dropout=0.3,
                           deviation=args.deviation, n_stim=100)
    print(param_hash(model))


if __name__ == "__main__":
    main()
