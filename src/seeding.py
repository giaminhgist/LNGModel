"""Single seeding source for the whole project.

`seed_all(seed)` must be called BEFORE any random operation: model
construction, stochastic preprocessing, sampler creation, or estimator
fitting. Each fold run reseeds with the same `run_seed`, making fold results
independent of execution order and of the previous fold's RNG state.

Determinism notes:
  - Python `hash()`/set iteration is NOT stabilized unless PYTHONHASHSEED is
    set by the launcher before Python starts (we do not rely on it: all
    iteration that affects results goes through seeded numpy/torch RNGs or
    sorted lists).
  - CPU/GPU bitwise equality is NOT promised (cuDNN, atomic reductions,
    BLAS kernels); determinism is guaranteed per device and per library
    version, up to float non-associativity within a device.
"""
import os
import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed python, numpy and torch (CPU and CUDA) from one integer."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic backend flags (best effort; does not promise bitwise
    # CPU/GPU equality, see module docstring).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def determinism_env_info() -> dict:
    """Environment info to record in every run (see run_info.json)."""
    import platform
    import sklearn
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "threads_omp": os.environ.get("OMP_NUM_THREADS"),
        "threads_mkl": os.environ.get("MKL_NUM_THREADS"),
        "threads_openblas": os.environ.get("OPENBLAS_NUM_THREADS"),
        "n_cpus": os.cpu_count(),
    }


# Files whose content determines run NUMERICS. Anything else (runners,
# summarizers, checks, reports) can be edited or added without invalidating
# completed runs. Adding a file NOT in this list must not change the hash,
# so the list is explicit rather than a directory scan.
_CODE_HASH_FILES = [
    "src/seeding.py",
    "src/metrics.py",
    "src/registry.py",
    "src/config.py",
    "src/rawdata.py",
    "src/scripts/preprocess.py",
    "src/data/common.py",
    "src/data/tabular.py",
    "src/features/stimulus.py",
    "src/features/subject.py",
    "src/models/proposal.py",
    "src/models/baselines.py",
    "src/training/config.py",
    "src/training/trainer.py",
    "pyproject.toml",
]


def code_hash():
    """Content hash of the code that determines run NUMERICS (explicit file
    whitelist above). Recorded in run_info.json so a resume/skip decision can
    detect code changes even between commits."""
    import hashlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for rel in _CODE_HASH_FILES:
        p = root / rel
        h.update(rel.encode())
        h.update(p.read_bytes())
    return h.hexdigest()
