"""Locked experiment registry for the current (2026-09-23) run matrix.

Single source of truth for: the 10 locked seeds, the 5 proposal configs
(RE-NUMBERED per user decision 2026-09-23) and the 10 baselines with their
subject-level representations.

ID mapping (re-numbering, recorded to avoid mixing methods across numbering):
  EXP-PROP-001 <- old EXP-PROP-006  mlp_deepset  (MAIN proposal)
  EXP-PROP-002 <- old EXP-PROP-005  mlp_mean
  EXP-PROP-003 <- old EXP-PROP-001  mlp_attn
  EXP-PROP-004 <- old EXP-PROP-007  z_mean (hard z-deviation)
  EXP-PROP-005 <- NEW               mlp_max (masked max pooling only)
Removed from the active suite: old 002 (lambda_norm — removed from code and
docs), old 003/004 (sub/zsub comparators), old 008/009 (diff/mahal hard
deviations; still implemented in the model but not run).

lambda_norm is REMOVED from the code and documentation (user decision
2026-09-23): the model has no HC-concentration regularization anymore.

Protocol: official subject-level 4-fold validation (Set_0..Set_3). Same folds
and same seeds for every experiment. Checkpoint policy (development
comparison): best outer-validation AUC epoch, earliest epoch wins ties; the
ACC/SEN/SPEC/F1 reported are taken at that same checkpoint.
"""
from training.config import PROTOCOL_VERSION

# Locked BEFORE any result was inspected. Do not add/remove/swap seeds based
# on scores.
SEEDS = [1234, 100, 0, 2024, 42, 3, 123, 13, 64, 2026]

FEATURE_SET = "full45"          # feature schema (all 45 base features)
N_FOLDS = 4
FOLD_NAMES = ["Set_0", "Set_1", "Set_2", "Set_3"]

# Proposal hyperparameters: max 150 epochs, patience 30, AdamW lr 1e-3,
# wd 1e-4, batch 16, dropout 0.3.
PROPOSAL_HP = dict(epochs=150, patience=30, lr=1e-3, weight_decay=1e-4,
                   batch_size=16, dropout=0.3)

# FNN hyperparameters: patience 20, same optimizer/lr/wd/batch/dropout,
# hidden 128-64-32.
FNN_HP = dict(epochs=150, patience=20, lr=1e-3, weight_decay=1e-4,
              batch_size=16, dropout=0.3)

# The 5 proposal configs (id, tag, deviation, comparator, pool)
PROPOSAL_EXPERIMENTS = [
    {"id": "EXP-PROP-001", "ablation": "mlp_deepset",
     "deviation": "learned", "comparator": "mlp", "pool": "deepset",
     "note": "MAIN proposal: learned latent deviation + mlp comparator + deepset (mean||max) pooling"},
    {"id": "EXP-PROP-002", "ablation": "mlp_mean",
     "deviation": "learned", "comparator": "mlp", "pool": "mean",
     "note": "mean pooling"},
    {"id": "EXP-PROP-003", "ablation": "mlp_attn",
     "deviation": "learned", "comparator": "mlp", "pool": "attention",
     "note": "attention pooling"},
    {"id": "EXP-PROP-004", "ablation": "z_mean",
     "deviation": "z", "comparator": None, "pool": "mean",
     "note": "hard z-deviation (fixed, no encoder/comparator/bank) + mean pooling"},
    {"id": "EXP-PROP-005", "ablation": "mlp_max",
     "deviation": "learned", "comparator": "mlp", "pool": "max",
     "note": "NEW: masked max pooling only"},
]

# Baselines: method -> subject-level representation (mapping preserved from
# the original baseline suite). agg = 2*45+1 (mean/std + n_valid_stim),
# catagg = 4*45+1 (per-category means + n_valid_stim), concat = 100*45.
# n_valid_stim is a derived counter, NOT one of the 45 base features.
BASELINE_METHODS = {
    "svm_rbf": "agg", "svm_linear": "agg", "rf": "agg", "qda": "agg",
    "gnb": "agg", "lr": "agg", "knn": "agg",
    "fnn": "agg", "fnn_cat": "catagg", "lr_l1": "concat",
}

PROTOCOL_VERSION = PROTOCOL_VERSION
