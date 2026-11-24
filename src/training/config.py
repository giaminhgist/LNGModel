"""Run configuration: typed dataclass + validated CLI parsing + run-dir naming.

Every run writes its full config to config.json inside the run directory:
  outputs/experiments/{experiment}/{feature_set}/seed{seed}/fold{fold}/config.json
Unknown CLI flags are REJECTED (no silent capture); model-specific options are
declared explicitly with allowed choices.
"""
import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from data.common import OUTPUTS

PROTOCOL_VERSION = "experiments-dev-1"
DEV_CHOICES = ("learned", "z", "diff", "mahal")
COMPARATOR_CHOICES = ("mlp", "sub", "zsub")
POOL_CHOICES = ("attention", "mean", "deepset", "max")


@dataclass
class RunConfig:
    # identity
    experiment: str = ""               # registry id, e.g. "EXP-PROP-001" / "fnn"
    feature_set: str = "full45"        # feature schema name (Phase 3: full45)
    protocol_version: str = PROTOCOL_VERSION
    proposal: str = "proposal"
    ablation: str = ""                 # short tag describing the variant
    seed: int = 42                     # run seed (reseeded at the start of each fold)
    fold: str = "Set_0"                # official fold name
    # training
    epochs: int = 150
    patience: int = 30                 # early stopping on val AUC
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    dropout: float = 0.3
    # model-specific options (validated by parse_args)
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        extra = d.pop("extra", {})
        cfg = cls(**d)
        cfg.extra = extra
        return cfg

    def run_dir(self, create=True):
        d = (OUTPUTS / "experiments" / self.experiment / self.feature_set /
             f"seed{self.seed}" / f"fold{self.fold}")
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, run_dir: Path):
        (run_dir / "config.json").write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, run_dir: Path):
        return cls.from_dict(json.loads((Path(run_dir) / "config.json").read_text()))


def parse_args(default_proposal="proposal"):
    """CLI for the proposal entry point. Unknown flags raise argparse errors."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="", help="registry id (run identity)")
    ap.add_argument("--feature_set", default="full45")
    ap.add_argument("--protocol_version", default=PROTOCOL_VERSION)
    ap.add_argument("--proposal", default=default_proposal)
    ap.add_argument("--ablation", default="base")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", default="Set_0")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--device", default="")
    # model-specific options (proposal)
    ap.add_argument("--deviation", choices=DEV_CHOICES, default="learned")
    ap.add_argument("--comparator", choices=COMPARATOR_CHOICES, default="mlp")
    ap.add_argument("--pool", choices=POOL_CHOICES, default="attention")
    args = ap.parse_args()
    extra = {"deviation": args.deviation, "comparator": args.comparator,
             "pool": args.pool}
    cfg = RunConfig(experiment=args.experiment, feature_set=args.feature_set,
                    protocol_version=args.protocol_version,
                    proposal=args.proposal, ablation=args.ablation,
                    seed=args.seed, fold=args.fold, epochs=args.epochs,
                    patience=args.patience, lr=args.lr,
                    weight_decay=args.weight_decay, batch_size=args.batch_size,
                    dropout=args.dropout, extra=extra)
    return cfg, args.device
